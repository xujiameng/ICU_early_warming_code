-- Materialized View: mimiciii.new_0709_vitalsfirstday1

-- DROP MATERIALIZED VIEW mimiciii.new_0709_vitalsfirstday1;


-- Description 
--   该程序功能为：计算患者进入ICU前12小时生理数据的衍生变量
--   程序流程为：选择符合要求的患者生理数据，并求解其最大值，最小值，平均值，方差，标准差，四分位数，极差。

--   程序运行结果；生成一个包含患者进入ICU 前12小时的生理数据的衍生变量的表




CREATE MATERIALIZED VIEW mimiciii.new_0709_vitalsfirstday1 AS 
 SELECT a.subject_id,
    a.hadm_id,
    a.icustay_id,
    b.heartrate_min,
    b.heartrate_max,
    b.heartrate_mean,
    b.heartrate_var,
    b.heartrate_stddev,
    b.heartrate_range,
    b.heartrate_perc_25,
    b.heartrate_perc_50,
    b.heartrate_perc_75,
    b.resprate_min,
    b.resprate_max,
    b.resprate_mean,
    b.resprate_var,
    b.resprate_stddev,
    b.resprate_range,
    b.resprate_perc_25,
    b.resprate_perc_50,
    b.resprate_perc_75,
    b.tempc_min,
    b.tempc_max,
    b.tempc_mean,
    b.tempc_var,
    b.tempc_stddev,
    b.tempc_range,
    b.tempc_perc_25,
    b.tempc_perc_50,
    b.tempc_perc_75,
    b.spo2_min,
    b.spo2_max,
    b.spo2_mean,
    b.spo2_var,
    b.spo2_stddev,
    b.spo2_range,
    b.spo2_perc_25,
    b.spo2_perc_50,
    b.spo2_perc_75,
    a.nisysbp_min,
    a.nisysbp_max,
    a.nisysbp_mean,
    a.nisysbp_var,
    a.nisysbp_stddev,
    a.nisysbp_range,
    a.nisysbp_perc_25,
    a.nisysbp_perc_50,
    a.nisysbp_perc_75,
    a.nidiasbp_min,
    a.nidiasbp_max,
    a.nidiasbp_mean,
    a.nidiasbp_var,
    a.nidiasbp_stddev,
    a.nidiasbp_range,
    a.nidiasbp_perc_25,
    a.nidiasbp_perc_50,
    a.nidiasbp_perc_75,
    a.nimeanbp_min,
    a.nimeanbp_max,
    a.nimeanbp_mean,
    a.nimeanbp_var,
    a.nimeanbp_stddev,
    a.nimeanbp_range,
    a.nimeanbp_perc_25,
    a.nimeanbp_perc_50,
    a.nimeanbp_perc_75
   FROM mimiciii.new_0709_ready_vitalsfirstday1 a
     LEFT JOIN mimiciii.new_vitalsfirstday1 b ON a.subject_id = b.subject_id AND a.hadm_id = b.hadm_id AND a.icustay_id = b.icustay_id
  ORDER BY a.subject_id, a.hadm_id, a.icustay_id
WITH DATA;

ALTER TABLE mimiciii.new_0709_vitalsfirstday1
  OWNER TO postgres;
