import numpy as np
from collections import defaultdict

def prepare_wave_data(waves_peak_info):
    waves_data = waves_peak_info.copy()
    
    point_types = [
        'ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets',
        'ECG_R_Onsets', 'ECG_Q_Peaks', 'ECG_R_Peaks', 
        'ECG_S_Peaks', 'ECG_R_Offsets',
        'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets'
    ]
    
    for point_type in point_types:
        if point_type in waves_data:
            points = np.array(waves_data[point_type], dtype=float)
            waves_data[point_type] = points[~np.isnan(points)]
    
    return waves_data

def group_pqrst_points(waves_data):
    all_points = []
    for point_type, points in waves_data.items():
        for point in points:
            all_points.append((point, point_type))
    
    all_points.sort(key=lambda x: x[0])
    point_order = {
        'ECG_P_Onsets': 0, 'ECG_P_Peaks': 1, 'ECG_P_Offsets': 2,
        'ECG_R_Onsets': 3, 'ECG_Q_Peaks': 4, 'ECG_R_Peaks': 5,
        'ECG_S_Peaks': 6, 'ECG_R_Offsets': 7,
        'ECG_T_Onsets': 8, 'ECG_T_Peaks': 9, 'ECG_T_Offsets': 10
    }
    
    cycles = []
    current_cycle = []
    last_point_order = -1
    
    for point_time, point_type in all_points:
        current_order = point_order.get(point_type, -1)
        if current_order < last_point_order:
            if current_cycle:
                cycles.append(current_cycle)
            current_cycle = []
            last_point_order = -1
        
        current_cycle.append((point_time, point_type))
        last_point_order = current_order
    
    if current_cycle:
        cycles.append(current_cycle)
    
    grouped_cycles = []
    for cycle in cycles:
        cycle_dict = defaultdict(list)
        for point_time, point_type in cycle:
            cycle_dict[point_type].append(point_time)
        grouped_cycles.append(cycle_dict)
    
    return grouped_cycles

def calculate_amplitudes(cycles, signal):
    for cycle in cycles:
        cycle_amplitudes = {}
        for point_type, times in cycle.items():
            if len(times) > 0:
                idx = int(np.round(times[0]))
                if 0 <= idx < len(signal):
                    cycle_amplitudes[f"{point_type}_amplitude"] = signal[idx]
        cycle['amplitudes'] = cycle_amplitudes
    return cycles

def calculate_intervals(cycles, fs):
    for cycle in cycles:
        cycle_intervals = {}
        point_types = list(cycle.keys())
        point_types = [pt for pt in point_types if pt not in ['amplitudes', 'intervals', 'diff_amplitudes']]
        
        for i, pt1 in enumerate(point_types):
            for j, pt2 in enumerate(point_types):
                if i >= j:
                    continue
                
                if len(cycle[pt1]) > 0 and len(cycle[pt2]) > 0:
                    interval_name = f"{pt1}_to_{pt2}"
                    
                    time_diff = (cycle[pt2][0] - cycle[pt1][0]) * 1000 / fs
                    cycle_intervals[f"{interval_name}_interval_ms"] = time_diff

        # QTc (корригированный QT по Базетту)
        if 'ECG_Q_Peaks' in cycle and 'ECG_T_Offsets' in cycle:
            if 'ECG_R_Peaks' in cycle and len(cycle['ECG_R_Peaks']) > 0:
                qt_interval = (cycle['ECG_T_Offsets'][0] - cycle['ECG_Q_Peaks'][0]) * 1000 / fs
                rr_interval = np.mean(np.diff(cycle['ECG_R_Peaks'])) * 1000 / fs if len(cycle['ECG_R_Peaks']) > 1 else 1000
                qtc = qt_interval / np.sqrt(rr_interval / 1000)
                cycle_intervals['QTc_interval_ms'] = qtc
        
        cycle['intervals'] = cycle_intervals
    
    return cycles

def calculate_amplitude_differences(cycles):
    for cycle in cycles:            
        amplitude_diffs = {}
        amplitudes = cycle['amplitudes']
        amplitude_keys = list(amplitudes.keys())
        
        for i, key1 in enumerate(amplitude_keys):
            for j, key2 in enumerate(amplitude_keys):
                if i >= j:
                    continue
                
                diff_name = f"{key1}_to_{key2}_diff"
                amplitude_diffs[diff_name] = abs(amplitudes[key1] - amplitudes[key2])
        cycle['diff_amplitudes'] = amplitude_diffs
    
    return cycles


"""def calculate_all_wave_areas(cycles, signal, fs):
    for cycle in cycles:
        areas = {}
        
        # Границы изолинии (базовый уровень)
        baseline = np.mean(signal[int(cycle.get('ECG_P_Onsets', [0])[0]-10*fs/1000):int(cycle.get('ECG_P_Onsets', [0])[0])])
        
        # Основные площади волн
        for wave in ['P', 'QRS', 'T']:
            onset_key = f'ECG_{wave}_Onsets' if wave != 'QRS' else 'ECG_Q_Peaks'
            offset_key = f'ECG_{wave}_Offsets' if wave != 'QRS' else 'ECG_S_Peaks'
            
            if onset_key in cycle and offset_key in cycle:
                start = int(cycle[onset_key][0])
                end = int(cycle[offset_key][0])
                segment = signal[start:end] - baseline
                
                # Общая площадь (с учётом знака)
                areas[f'{wave}_area'] = np.trapz(segment, dx=1/fs)
                
                # Абсолютная площадь (без учёта знака)
                areas[f'{wave}_abs_area'] = np.trapz(np.abs(segment), dx=1/fs)

                areas[f'{wave}_area_above'] = np.trapz(np.maximum(segment, 0), dx=1/fs)
                areas[f'{wave}_area_below'] = np.trapz(np.minimum(segment, 0), dx=1/fs)
        
        # Площади подсегментов QRS
        if 'ECG_Q_Peaks' in cycle and 'ECG_R_Peaks' in cycle and 'ECG_S_Peaks' in cycle:
            q = int(cycle['ECG_Q_Peaks'][0])
            r = int(cycle['ECG_R_Peaks'][0])
            s = int(cycle['ECG_S_Peaks'][0])
            
            areas['Q_R_area'] = np.trapz(signal[q:r] - baseline, dx=1/fs)
            areas['R_S_area'] = np.trapz(signal[r:s] - baseline, dx=1/fs)
        
        # Площадь ST-сегмента (от S до T)
        if 'ECG_S_Peaks' in cycle and 'ECG_T_Onsets' in cycle:
            s = int(cycle['ECG_S_Peaks'][0])
            t_start = int(cycle['ECG_T_Onsets'][0])
            areas['ST_area'] = np.trapz(signal[s:t_start] - baseline, dx=1/fs)
        
        cycle['areas'] = areas
    return cycles"""


import numpy as np
from collections import defaultdict

def find_nearest_crossing(signal, point, baseline, direction='left', limit=None):
    """Находит ближайшее пересечение с базовой линией с обработкой ошибок"""
    try:
        if direction == 'left':
            start = point - 1
            end = 0 if limit is None else max(limit, 0)
            step = -1
        else:
            start = point + 1
            end = len(signal)-1 if limit is None else min(limit, len(signal)-1)
            step = 1
        
        for i in range(start, end, step):
            if (signal[i] - baseline) * (signal[i-step] - baseline) <= 0:
                return i
        return None
    except Exception as e:
        print(f"Error in find_nearest_crossing: {str(e)}")
        return None

def calculate_qrs_area_with_dynamic_baseline(signal, q_point, r_point, s_point, fs, baseline_window=50):
    """Расчет площади QRS с обработкой ошибок"""
    try:
        baseline_samples = int(baseline_window * fs / 1000)
        baseline_start = max(0, q_point - baseline_samples)
        baseline = np.median(signal[baseline_start:q_point])
        
        r_amplitude = signal[r_point] - baseline
        q_amplitude = signal[q_point] - baseline
        is_positive_direction = (r_amplitude - q_amplitude) > 0
        
        def safe_trapz(segment, is_above):
            try:
                if is_above:
                    return abs(np.trapz(np.maximum(segment, 0), dx=1/fs))
                return abs(np.trapz(np.minimum(segment, 0), dx=1/fs))
            except:
                return 0.0
        
        qr_area = safe_trapz(signal[q_point:r_point+1] - baseline, is_positive_direction)
        rs_area = safe_trapz(signal[r_point:s_point+1] - baseline, not is_positive_direction)
        
        return {
            'qrs_dynamic_area': qr_area + rs_area,
            'qrs_dynamic_qr_area': qr_area,
            'qrs_dynamic_rs_area': rs_area,
        }
    except Exception as e:
        print(f"Error in calculate_qrs_area: {str(e)}")
        return {
            'qrs_dynamic_area': 0.0,
            'qrs_dynamic_qr_area': 0.0,
            'qrs_dynamic_rs_area': 0.0,
        }

def calculate_all_wave_areas(cycles, signal, fs):
    """Основная функция с полной обработкой ошибок"""
    for cycle in cycles:
        try:
            areas = defaultdict(float)
            
            # 1. Базовая линия
            try:
                baseline_window = int(10 * fs / 1000)
                p_onset = int(cycle.get('ECG_P_Onsets', [0])[0])
                baseline_start = max(0, p_onset - baseline_window)
                baseline = np.median(signal[baseline_start:p_onset])
            except:
                baseline = 0.0
            
            # 2. Направление QRS
            qrs_direction = None
            try:
                if all(k in cycle for k in ['ECG_Q_Peaks', 'ECG_R_Peaks']):
                    q = int(cycle['ECG_Q_Peaks'][0])
                    r = int(cycle['ECG_R_Peaks'][0])
                    qrs_direction = 'positive' if (signal[r] - baseline) > (signal[q] - baseline) else 'negative'
            except:
                qrs_direction = 'unknown'
            
            # 3. P-wave
            try:
                if 'ECG_P_Onsets' in cycle and 'ECG_P_Offsets' in cycle:
                    p_start = int(cycle['ECG_P_Onsets'][0])
                    p_end = int(cycle['ECG_P_Offsets'][0])
                    if p_start < p_end:
                        p_segment = signal[p_start:p_end] - baseline
                        areas['P_area'] = np.trapz(p_segment, dx=1/fs)
                        areas['P_abs_area'] = np.trapz(np.abs(p_segment), dx=1/fs)
            except Exception as e:
                print(f"P-wave calculation error: {str(e)}")
            
            # 4. T-wave (оба варианта)
            try:
                if 'ECG_T_Onsets' in cycle and 'ECG_T_Offsets' in cycle:
                    t_onset = int(cycle['ECG_T_Onsets'][0])
                    t_offset = int(cycle['ECG_T_Offsets'][0])
                    
                    # Вариант 1: между точками пересечения
                    t_left = find_nearest_crossing(signal, t_onset, baseline, 'left')
                    t_right = find_nearest_crossing(signal, t_offset, baseline, 'right')
                    
                    t_start = t_left if t_left is not None else t_onset
                    t_end = t_right if t_right is not None else t_offset
                    
                    if t_start < t_end:
                        t_segment = signal[t_start:t_end] - baseline
                        areas['T_wave_cross_area'] = np.trapz(t_segment, dx=1/fs)
                        areas['T_wave_cross_duration'] = (t_end - t_start) * 1000 / fs
                    
                    # Вариант 2: стандартный
                    if t_onset < t_offset:
                        t_segment = signal[t_onset:t_offset] - baseline
                        areas['T_area'] = np.trapz(t_segment, dx=1/fs)
                        if qrs_direction in ['positive', 'negative']:
                            areas['T_area_corrected'] = -areas['T_area'] if qrs_direction == 'negative' else areas['T_area']
            except Exception as e:
                print(f"T-wave calculation error: {str(e)}")
            
            # 5. QRS комплекс
            try:
                if all(k in cycle for k in ['ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks']):
                    q = int(cycle['ECG_Q_Peaks'][0])
                    r = int(cycle['ECG_R_Peaks'][0])
                    s = int(cycle['ECG_S_Peaks'][0])
                    
                    # Динамический расчет
                    qrs_dynamic = calculate_qrs_area_with_dynamic_baseline(signal, q, r, s, fs)
                    areas.update(qrs_dynamic)
                    
                    # Стандартный расчет
                    if q < s:
                        qrs_segment = signal[q:s+1] - baseline
                        areas['QRS_area'] = np.trapz(qrs_segment, dx=1/fs)
                        
                        # Подсегменты
                        if q < r:
                            areas['Q_R_area'] = np.trapz(signal[q:r] - baseline, dx=1/fs)
                        if r < s:
                            areas['R_S_area'] = np.trapz(signal[r:s] - baseline, dx=1/fs)
            except Exception as e:
                print(f"QRS calculation error: {str(e)}")
            
            # 6. S-wave area
            try:
                if all(k in cycle for k in ['ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Onsets']):
                    r = int(cycle['ECG_R_Peaks'][0])
                    s = int(cycle['ECG_S_Peaks'][0])
                    t_onset = int(cycle['ECG_T_Onsets'][0])
                    
                    s_left = find_nearest_crossing(signal, s, baseline, 'left', r)
                    s_right = find_nearest_crossing(signal, s, baseline, 'right', t_onset)
                    
                    s_start = s_left if s_left is not None else r
                    s_end = s_right if s_right is not None else t_onset
                    
                    if s_start < s_end:
                        s_segment = signal[s_start:s_end] - baseline
                        areas['S_area'] = np.trapz(s_segment, dx=1/fs)
                        areas['S_duration_ms'] = (s_end - s_start) * 1000 / fs
            except Exception as e:
                print(f"S-wave calculation error: {str(e)}")
            
            cycle['areas'] = dict(areas)
            
        except Exception as e:
            print(f"Error processing cycle: {str(e)}")
            cycle['areas'] = {}
    
    return cycles

def calculate_area_ratios(cycles):
    for cycle in cycles:
        if 'areas' not in cycle:
            continue
            
        ratios = {}
        areas = cycle['areas']
        
        wave_pairs = [('T', 'QRS'), ('P', 'QRS'), ('T', 'P'), ('Q_R', 'R_S')]
        
        for (wave1, wave2) in wave_pairs:
            key1 = f'{wave1}_area'
            key2 = f'{wave2}_area'
            
            if key1 in areas and key2 in areas and areas[key2] != 0:
                ratio = areas[key1] / areas[key2]
                ratios[f'{wave1}_{wave2}_ratio'] = ratio
        
        if 'T_area' in areas and 'QRS_abs_area' in areas and areas['QRS_abs_area'] != 0:
            ratios['T_QRSabs_ratio'] = areas['T_area'] / areas['QRS_abs_area']
        
        if 'ST_area' in areas and 'QRS_area' in areas and areas['QRS_area'] != 0:
            ratios['ST_QRS_ratio'] = areas['ST_area'] / areas['QRS_area']
        
        cycle['area_ratios'] = ratios
    return cycles


def calc_average_signal(signal, r_peaks, fs):
    before_r = int(0.3 * fs)  # 300 мс до R
    after_r = int(0.5 * fs)   # 500 мс после R
    segments = []
    
    for r in r_peaks:
        start = r - before_r
        end = r + after_r
        if start >= 0 and end <= len(signal):
            segments.append(signal[start:end])
    
    return np.mean(segments, axis=0), before_r, after_r


def compute_stats(values, prefix="", only_median=False):
    clean_values = values[~np.isnan(values)]
    
    if len(clean_values) == 0:
        stats = {
            f"{prefix}_median": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
        return {k: stats[k] for k in [f"{prefix}_median"]} if only_median else stats

    stats = {}
    stats[f"{prefix}_median"] = np.median(clean_values)
    
    if not only_median:
        stats[f"{prefix}_mean"] = np.mean(clean_values)
        stats[f"{prefix}_std"] = np.std(clean_values)
        stats[f"{prefix}_q25"] = np.quantile(clean_values, 0.25)
        stats[f"{prefix}_q75"] = np.quantile(clean_values, 0.75)
        stats[f"{prefix}_min"] = np.min(clean_values)
        stats[f"{prefix}_max"] = np.max(clean_values)
    
    return stats

def calculate_statistics(grouped_cycles, only_median=False):
    feature_categories = ['amplitudes', 'intervals', 'diff_amplitudes', 'areas', 'area_ratios']
    all_stats = {}
    
    for category in feature_categories:
        category_data = defaultdict(list)
        for cycle in grouped_cycles:
            if category in cycle:
                for feature, value in cycle[category].items():
                    category_data[feature].append(value)
        
        for feature, values in category_data.items():
            values = np.array(values)
            #print(feature, values)
            clean_values = values[~np.isnan(values)]
            if len(clean_values) == 0:
                continue
                
            prefix = f"{feature}"
            if only_median:
                all_stats[f"{prefix}_median"] = np.median(clean_values)
            else:   
                all_stats[f"{prefix}_median"] = np.median(clean_values)
                all_stats[f"{prefix}_mean"] = np.mean(clean_values)
                all_stats[f"{prefix}_std"] = np.std(clean_values)
                all_stats[f"{prefix}_q25"] = np.quantile(clean_values, 0.25)
                all_stats[f"{prefix}_q75"] = np.quantile(clean_values, 0.75)
                all_stats[f"{prefix}_min"] = np.min(clean_values)
                all_stats[f"{prefix}_max"] = np.max(clean_values)
    
    return all_stats