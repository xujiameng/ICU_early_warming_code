
-- Materialized View: mimiciii.new_0709_ready_vitalsfirstday2

-- DROP MATERIALIZED VIEW mimiciii.new_0709_ready_vitalsfirstday2;



-- Description 
--   该程序功能为：计算患者进入ICU 12-24小时无创血压时序数据的衍生变量
--   程序流程为：选择符合要求的患者血压数据，并求解其最大值，最小值，平均值，方差，标准差，四分位数，极差。

--   程序运行结果；生成一个包含患者进入ICU 12-24小时的血压数据的衍生变量的表


CREATE MATERIALIZED VIEW mimiciii.new_0709_ready_vitalsfirstday2 AS 
 SELECT pvt.subject_id,               -- 计算时序数据的大值，最小值，平均值，方差，标准差，四分位数，极差。
    pvt.hadm_id,
    pvt.icustay_id,
    min(pvt.nisysbp) AS nisysbp_min,
    max(pvt.nisysbp) AS nisysbp_max,
    avg(pvt.nisysbp) AS nisysbp_mean,
    variance(pvt.nisysbp) AS nisysbp_var,
    stddev(pvt.nisysbp) AS nisysbp_stddev,
    max(pvt.nisysbp) - min(pvt.nisysbp) AS nisysbp_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (pvt.nisysbp::double precision)) AS nisysbp_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (pvt.nisysbp::double precision)) AS nisysbp_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (pvt.nisysbp::double precision)) AS nisysbp_perc_75,
    min(pvt.nidiasbp) AS nidiasbp_min,
    max(pvt.nidiasbp) AS nidiasbp_max,
    avg(pvt.nidiasbp) AS nidiasbp_mean,
    variance(pvt.nidiasbp) AS nidiasbp_var,
    stddev(pvt.nidiasbp) AS nidiasbp_stddev,
    max(pvt.nidiasbp) - min(pvt.nidiasbp) AS nidiasbp_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (pvt.nidiasbp::double precision)) AS nidiasbp_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (pvt.nidiasbp::double precision)) AS nidiasbp_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (pvt.nidiasbp::double precision)) AS nidiasbp_perc_75,
    min(pvt.nimeanbp) AS nimeanbp_min,
    max(pvt.nimeanbp) AS nimeanbp_max,
    avg(pvt.nimeanbp) AS nimeanbp_mean,
    variance(pvt.nimeanbp) AS nimeanbp_var,
    stddev(pvt.nimeanbp) AS nimeanbp_stddev,
    max(pvt.nimeanbp) - min(pvt.nimeanbp) AS nimeanbp_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (pvt.nimeanbp::double precision)) AS nimeanbp_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (pvt.nimeanbp::double precision)) AS nimeanbp_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (pvt.nimeanbp::double precision)) AS nimeanbp_perc_75
   FROM ( SELECT ccc.subject_id,
            ccc.hadm_id,
            ccc.icustay_id,
                CASE
                    WHEN ccc.nisysbp::double precision > 0::double precision AND ccc.nisysbp::double precision < 400::double precision THEN ccc.nisysbp -- 剔除异常值
                    ELSE NULL::numeric
                END AS nisysbp,
                CASE
                    WHEN ccc.nidiasbp::double precision > 0::double precision AND ccc.nidiasbp::double precision < 300::double precision THEN ccc.nidiasbp
                    ELSE NULL::numeric
                END AS nidiasbp,
                CASE
                    WHEN ccc.nimeanbp::double precision > 0::double precision AND ccc.nimeanbp::double precision < 300::double precision THEN ccc.nimeanbp
                    ELSE NULL::numeric
                END AS nimeanbp
           FROM mimiciii.rf0709_h12_runin ccc) pvt
  GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id
  ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id
WITH DATA;

ALTER TABLE mimiciii.new_0709_ready_vitalsfirstday2
  OWNER TO postgres;
