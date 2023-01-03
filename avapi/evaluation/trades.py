
import pickle
from tqdm import tqdm
from copy import copy, deepcopy
import numpy as np

import avstack
from avstack.datastructs import DataContainer
from avstack.modules.perception.detections import BoxDetection

from avapi.evaluation.perception import get_percep_results_from_folder
from avapi.evaluation.tracking import TrackEvaluator, AvstackTrackDataset, \
    get_track_results_from_folder
from avapi.evaluation.tracking import HOTA, CLEAR, VACE, IDEucl
from avapi.evaluation.prediction import get_predict_results_from_folder


partial_occ = avstack.objects.Occlusion.PARTIAL

whitelist_types = {'kitti':['Car', 'Cyclist'],
                   'nuscenes':['car', 'bicycle',
                            'truck', 'bus', 'motorcycle'],
                   'carla':['car', 'truth', 'bicycle', 'motorcycle']}

ds_to_fr = {'kitti':10, 'nuscenes':2, 'carla':10}


def init_AV(AVs, i_case, dataset, li_perception, ca_perception, tracking, **kwargs):
    # -- pull off parameters
    AV = AVs[i_case] if isinstance(AVs, dict) else AVs
    li_p_in = li_perception[i_case] if isinstance(li_perception, dict) else li_perception
    ca_p_in = ca_perception[i_case] if isinstance(ca_perception, dict) else ca_perception
    trk_in  = tracking[i_case] if isinstance(tracking, dict) else tracking
    # -- init AV
    AV = AV(t_init=0,
             ego_init=None,
             lidar_perception=li_p_in,
             camera_perception=ca_p_in,
             tracking=trk_in,
             dataset=dataset.lower(),
             framerate=ds_to_fr[dataset.lower()],
             gpu=0,
             save_output=True,
             save_folder='results',
             **kwargs)
    return AV

# =======================================================
# TRADE STUDIES
# =======================================================

def run_trades(SMs, AVs, li_perception, ca_perception, tracking,
        n_cases_max=None, case_list_run=None,
        n_trials_max=10, trial_indices=None, frame_start=2, max_frames=20,
        max_framerate=None, max_dist=60, max_lidar_range=None,
        filter_front=True, sensor_eval='main_lidar', sensor_eval_super=None,
        trade_type='standard', save_result=True,
        save_file_base='study-1-{}-seq-res.p', print_results=True, **kwargs):
    """
    Wrapper around standard case for trade studies
    """
    frame_res_all = []
    seq_res_all = []
    if case_list_run is None:
        all_cases = list(AVs.keys())
    else:
        all_cases = case_list_run
    if n_cases_max is not None:
        all_cases = all_cases[:n_cases_max]

    for SM in SMs:
        per_frame_results = []
        per_seq_results = []
        n_trials = min(len(SM), n_trials_max)
        if trial_indices is None:
            trials = range(n_trials)
        else:
            trials = trial_indices[:n_trials]
        print('Running dataset {} over {} trials'.format(SM.name, n_trials))
        for i, i_trial in enumerate(trials):
            print('   Running trial {}, using index {}'.format(i, i_trial))
            SD = SM.get_scene_dataset_by_index(i_trial)
            for i_case in all_cases:
                # -- break out kwargs by case, if applicable
                kwargs_in = {}
                for k, v in kwargs.items():
                    if isinstance(v, dict):
                        kwargs_in[k] = v[i_case]
                    else:
                        kwargs_in[k] = v

                # -- break out trade type, if needed
                if isinstance(trade_type, dict):
                    trade_in = trade_type[i_case]
                else:
                    trade_in = trade_type

                # -- break out eval frame, if needed
                if isinstance(sensor_eval, dict):
                    sensor_eval_in = sensor_eval[i_case]
                else:
                    sensor_eval_in = sensor_eval
                if sensor_eval_super is None:
                    sensor_eval_super_in = sensor_eval_in
                else:
                    sensor_eval_super_in = sensor_eval_super

                # -- break outt filter front if needed
                if isinstance(filter_front, dict):
                    filter_front_in = filter_front[i_case]
                else:
                    filter_front_in = filter_front

                # -- run sequence
                AV = init_AV(AVs, i_case, SD.name, li_perception, ca_perception, tracking, **kwargs_in)
                print('      Running dataset: {}, case {}'.format(SD.name, i_case))
                AV, frame_results, seq_results = run_case(
                    SD, AV, i_trial, i_case, frame_start=frame_start, max_framerate=max_framerate,
                    max_frames=max_frames, max_dist=max_dist, max_lidar_range=max_lidar_range,
                    filter_front=filter_front_in, trade_type=trade_in,
                    sensor_eval=sensor_eval_in, sensor_eval_super=sensor_eval_super,
                    **kwargs_in)

                # -- store results
                if print_results:
                    print(seq_results)
                per_frame_results.append(frame_results)
                per_seq_results.append(seq_results)

        # save per-dataset results
        if save_result:
            seq_res_file = save_file_base.format(SM.name.lower())
            with open(seq_res_file, 'wb') as f:
                pickle.dump(per_seq_results, f)

        # store
        frame_res_all.append(per_frame_results)
        seq_res_all.append(per_seq_results)
    return frame_res_all, seq_res_all


def run_case(SD, AV, i_trial, i_case, frame_start, max_framerate, max_frames,
        max_dist, max_lidar_range, filter_front, trade_type, sensor_eval,
        sensor_eval_super, **kwargs):
    """
    Runs a standard trade study case
    """
    # --- assign tick function
    if trade_type == 'standard':
        tick_func = tick_standard
    elif trade_type == 'collaborative':
        tick_func = tick_collaborative
    else:
        raise NotImplementedError(trade_type)
    frame_results = []
    data_recur = {}
    data_manager = avstack.datastructs.DataManager(max_size=5)
    t_last = -np.inf
    for frame in tqdm(SD.frames[frame_start:min(max_frames+frame_start,len(SD.frames))]):
        # --- check framerate:
        if max_framerate is None:
            pass
        else:
            t_curr = SD.get_timestamp(frame, sensor_eval)
            if (t_curr - t_last) < (1./max_framerate - 1e-5):
                continue
            t_last = t_curr

        # -- run tick
        data_recur = tick_func(data_recur, AV, data_manager, SD, frame, filter_front, max_lidar_range, **kwargs)

        # -- store results
        res_frame = {'Case':i_case, 'Dataset':SD.name, 'Trial':i_trial, 'Frame':frame}
        # -- add metrics from data_recur
        if 'metrics' in data_recur:
            res_frame.update(data_recur['metrics'])

        frame_results.append(res_frame)

    # -- store aggregate results
    c_res = get_agg_collaborative_metrics(SD, AV, frame_results)
    p_res = get_agg_percep_metrics(SD, AV, max_dist=max_dist,
        sensor_eval=sensor_eval, sensor_eval_super=sensor_eval_super)
    for k in p_res[0]:
        for frame, res_frame in zip(p_res[0][k], frame_results):
            res_frame['Result_{}'.format(k)] = p_res[0][k][frame]['result']
    t_res = get_agg_tracking_metrics(SD, AV, max_dist=max_dist,
         sensor_eval=sensor_eval, sensor_eval_super=sensor_eval_super)
    for frame, res_frame in zip(t_res[0], frame_results):
        res_frame['Result_{}'.format('tracking')] = t_res[0][frame]['result']


    metrics = {'perception':p_res[1],
               'tracking':t_res[1],
               'prediction':get_agg_prediction_metrics(SD, AV,
                    sensor_eval=sensor_eval, sensor_eval_super=sensor_eval_super)}
    if len(c_res) > 0:
        metrics['collaborative'] = c_res

    # -- expand all metrics
    metrics_exp = {'Metrics_{}_{}'.format(k1,k2):v2 for k1 in metrics for k2, v2 in metrics[k1].items()}
    seq_results = {'Case':i_case, 'Dataset':SD.name, 'Trial':i_trial}
    seq_results.update(metrics_exp)

    return AV, frame_results, seq_results

# ------------------------------------------------------------------------------

def tick_standard(data_recur, AV, data_manager, SD, frame, filter_front, max_lidar_range, **kwargs):
    # -- sensor data
    data_manager.push(SD.get_lidar(frame, sensor='main_lidar', filter_front=filter_front, max_range=max_lidar_range))  # defaults to main lidar
    data_manager.push(SD.get_image(frame, sensor='main_camera'))  # defaults to main camera
    # -- run AV
    AV.tick(frame=frame, data_manager=data_manager, timestamp=None)
    return data_recur


def tick_collaborative(data_recur, AV, data_manager, SD, frame, filter_front, max_lidar_range,
        collaborative_sensors, collaborative_source, collaborative_range,
        collaborative_noise, collaborative_rate, **kwargs):

    # -- standard sensor data
    data_manager.push(SD.get_lidar(frame, sensor='main_lidar', filter_front=filter_front, max_range=max_lidar_range))  # defaults to main lidar
    data_manager.push(SD.get_image(frame, sensor='main_camera'))  # defaults to main camera

    # -- infrastructure data for all sensors
    default_collab = {'t_last_send':-np.inf, 'last_frame_available':None, 'last_frame_sent':None}
    data_recur['metrics'] = {}
    data_recur['metrics']['collab_detections'] = []
    data_recur['metrics']['collab_sensor_ids_in_view'] = []
    data_recur['metrics']['collab_sensor_ids_made_dets'] = []
    data_recur['metrics']['n_collab_sensors_total'] = len(collaborative_sensors)
    data_recur['metrics']['n_collab_sensors_in_view'] = 0
    data_recur['metrics']['n_collab_detections'] = 0
    for sensor in collaborative_sensors:
        # -- get current time and preallocate on first round
        curr_time = SD.get_timestamp(frame, sensor=sensor)
        if sensor not in data_recur:
            data_recur[sensor] = deepcopy(default_collab)

        # -- always get the last available frame so we can be ready to send
        # -- also check if in range of the ego
        # NOTE: we assume the sensor calib is in ego frame
        try:
            calib = SD.get_calibration(frame, sensor=sensor)
            if np.linalg.norm(calib.origin.x) > collaborative_range:
                continue
        except Exception as e:
            raise e  # figure out what exception this would be....
            continue
        else:
            data_recur[sensor]['last_frame_available'] = frame
            data_recur['metrics']['n_collab_sensors_in_view'] += 1
            data_recur['metrics']['collab_sensor_ids_in_view'].append(int(sensor[-3:]))

        # -- check if its time to make a measurement
        if (curr_time - data_recur[sensor]['t_last_send']) < 1./collaborative_rate-1e-4:
            continue  # not ready to make measurement

        # -- convert the raw sensor data to what we want
        if data_recur[sensor]['last_frame_available'] == data_recur[sensor]['last_frame_sent']:
            raise RuntimeError('Asking to send duplicate frames...check data rates {}'.format(data_recur[sensor]))

        c_frame = data_recur[sensor]['last_frame_available']
        c_timestamp = SD.get_timestamp(c_frame, sensor=sensor)
        if collaborative_source == 'ground_truth':
            # -- get truth
            objects = SD.get_objects(c_frame, sensor=sensor)
            boxes = [obj.box3d for obj in objects]
            obj_types = [obj.obj_type for obj in objects]
            for box in boxes:
                if box.where_is_t != 'center':
                    box.center_box()
                if collaborative_noise is not None:
                    box.add_noise(collaborative_noise)
            scores = [1.0] * len(boxes)
        elif collaborative_source == 'camera-to-3d':
            # -- run a real perception algorithm
            raise NotImplementedError
            if 'perception_algorithm' not in data_recur:
                data_recur['perception_algorithm'] = avstack
        else:
            raise NotImplementedError(collaborative_source)
        detections = [BoxDetection(sensor, box, obj_type, score) for box,
            obj_type, score in zip(boxes, obj_types, scores)]
        collab_data = DataContainer(c_frame, c_timestamp, detections, 'collaborative_'+sensor)
        data_manager.push(collab_data)

        # -- save recur results
        data_recur['metrics']['collab_detections'] += detections
        data_recur['metrics']['n_collab_detections'] += len(detections)
        data_recur['metrics']['collab_sensor_ids_made_dets'].append(int(sensor[-3:]))
        data_recur[sensor]['t_last_send'] = curr_time
        data_recur[sensor]['last_frame_sent'] = c_frame

    # -- run AV
    AV.tick(frame=frame, data_manager=data_manager, timestamp=None)

    return data_recur



# =======================================================
# AGGREGATE EVALUATIONS
# =======================================================

def get_agg_collaborative_metrics(SD, AV, frame_results):
    key_list = ['n_collab_detections', 'n_collab_sensors_in_view', 'n_collab_sensors_total']
    c_res = {}
    k_agg = '{}_median'
    for res in frame_results:
        for k in key_list:
            if k in res:
                if k_agg.format(k) not in c_res:
                    c_res[k_agg.format(k)] = []
                c_res[k_agg.format(k)].append(res[k])
    for k, v in c_res.items():
        c_res[k] = np.median(v)
    return c_res


def get_agg_percep_metrics(SD, AV, sensor_eval, sensor_eval_super='main_lidar',
        max_dist=80, max_occ=partial_occ, multiprocess=False):
    # the 0 index is to get per frame results
    res = {k:get_percep_results_from_folder(SD, result_path=v.save_folder, sensor_eval=sensor_eval,
                                            sensor_eval_super=sensor_eval_super,
                                            max_dist=max_dist, max_occ=max_occ,
                                            whitelist_types=whitelist_types[SD.name.lower()],
                                            multiprocess=multiprocess) for k, v in AV.perception.items()}
    per_frame_res = {k:res[k][0] for k, v in AV.perception.items()}

    # the 1 index is to get aggregate results only
    per_seq_res = {k:res[k][1] for k, v in AV.perception.items()}
    per_seq_res = {'{}_{}'.format(k1, k2):v2 for k1 in per_seq_res for k2, v2 in per_seq_res[k1].items()}
    return per_frame_res, per_seq_res


def get_agg_tracking_metrics(SD, AV, sensor_eval, sensor_eval_super='main_lidar',
        trk_name='box-tracker', objects='all_objects', max_dist=80, max_occ=partial_occ,
        multiprocess=False):
    """Get aggregate metrics"""
    # --- per-frame analysis
    per_frame_res, per_seq_res = get_track_results_from_folder(
        SD, result_path=AV.tracking.save_folder,
        sensor_eval=sensor_eval, sensor_eval_super=sensor_eval_super, max_dist=max_dist,
        max_occ=max_occ, whitelist_types=whitelist_types[SD.name.lower()], multiprocess=multiprocess)

    # --- metrics evaluator
    eval_config = TrackEvaluator.get_default_eval_config()
    eval_config['PRINT_CONFIG'] = False
    eval_config['PRINT_RESULTS'] = False
    evaluator = TrackEvaluator(eval_config)
    ds = AvstackTrackDataset(SD, AV.tracking.save_folder, trk_name, sensor_eval=sensor_eval,
         sensor_eval_super=sensor_eval_super, whitelist_types=whitelist_types[SD.name.lower()],
         max_dist=max_dist)
    eval_results = evaluator.evaluate([ds],
                                      [HOTA(), CLEAR(), VACE(), IDEucl()])

    # Expand metrics
    mets = eval_results[0]['AvstackTrackDataset'][trk_name][SD.sequence_id][objects]
    mets = {'{}_{}'.format(k1,k2):v2 for k1 in mets for k2, v2 in mets[k1].items()}
    return per_frame_res, mets


def get_agg_prediction_metrics(SD, AV, sensor_eval, sensor_eval_super='main_lidar',
        multiprocess=False):
    """Get aggregate and per-frame metrics"""
    pred_res, pred_agg = get_predict_results_from_folder(
        SD, result_path=AV.prediction.save_folder, sensor_eval=sensor_eval, sensor_eval_super=sensor_eval_super,
            whitelist_types=whitelist_types[SD.name.lower()], multiprocess=multiprocess)
    return pred_agg
