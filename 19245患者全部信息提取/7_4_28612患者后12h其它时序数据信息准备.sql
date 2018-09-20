-- Materialized View: mimiciii.new_vitalsfirstday2

-- DROP MATERIALIZED VIEW mimiciii.new_vitalsfirstday2;


-- Description 

--   该程序功能为：计算患者进入ICU 12~24 小时除无创血压外其它时序数据的衍生变量
--   程序流程为：选择符合要求的患者时序数据，并求解其最大值，最小值，平均值，方差，标准差，四分位数，极差。

--   程序运行结果；生成一个包含患者进入ICU 12~24 小时的时序的衍生变量的表



CREATE MATERIALIZED VIEW mimiciii.new_vitalsfirstday2 AS 
 SELECT pvt.subject_id,   	-- 计算最大值，最小值，平均值，方差，标准差，四分位数，极差等
    pvt.hadm_id,
    pvt.icustay_id,
    min(
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS heartrate_min,
    max(
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS heartrate_max,
    avg(
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS heartrate_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS heartrate_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS heartrate_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS heartrate_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS heartrate_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS heartrate_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 1 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS heartrate_perc_75,
    min(
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp_min,
    max(
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp_max,
    avg(
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS sysbp_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS sysbp_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 2 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS sysbp_perc_75,
    min(
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp_min,
    max(
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp_max,
    avg(
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS diasbp_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS diasbp_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 3 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS diasbp_perc_75,
    min(
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp_min,
    max(
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp_max,
    avg(
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS meanbp_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS meanbp_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 4 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS meanbp_perc_75,
    min(
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS resprate_min,
    max(
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS resprate_max,
    avg(
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS resprate_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS resprate_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS resprate_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS resprate_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS resprate_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS resprate_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 5 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS resprate_perc_75,
    min(
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tempc_min,
    max(
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tempc_max,
    avg(
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tempc_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tempc_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tempc_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tempc_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS tempc_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS tempc_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 6 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS tempc_perc_75,
    min(
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2_min,
    max(
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2_max,
    avg(
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS spo2_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS spo2_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 7 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS spo2_perc_75,
    min(
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nisysbp_min,
    max(
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nisysbp_max,
    avg(
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nisysbp_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nisysbp_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nisysbp_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nisysbp_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS nisysbp_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS nisysbp_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 8 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS nisysbp_perc_75,
    min(
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nidiasbp_min,
    max(
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nidiasbp_max,
    avg(
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nidiasbp_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nidiasbp_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nidiasbp_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nidiasbp_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS nidiasbp_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS nidiasbp_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 9 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS nidiasbp_perc_75,
    min(
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nimeanbp_min,
    max(
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nimeanbp_max,
    avg(
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nimeanbp_mean,
    variance(
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nimeanbp_var,
    stddev(
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nimeanbp_stddev,
    max(
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END) - min(
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nimeanbp_range,
    percentile_cont(0.25::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS nimeanbp_perc_25,
    percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS nimeanbp_perc_50,
    percentile_cont(0.75::double precision) WITHIN GROUP (ORDER BY (
        CASE
            WHEN pvt.vitalid = 10 THEN pvt.valuenum
            ELSE NULL::double precision
        END)) AS nimeanbp_perc_75
   FROM ( SELECT ie.subject_id,
            ie.hadm_id,
            ie.icustay_id,
                CASE       --根据官网提供的代码编号，对不同数据变量重新进行编号
                    WHEN (ce.itemid = ANY (ARRAY[211, 220045])) AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN 1
                    WHEN (ce.itemid = ANY (ARRAY[51, 6701, 220050])) AND ce.valuenum > 0::double precision AND ce.valuenum < 400::double precision THEN 2
                    WHEN (ce.itemid = ANY (ARRAY[8368, 8555, 220051])) AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN 3
                    WHEN (ce.itemid = ANY (ARRAY[52, 6702, 220052, 225312])) AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN 4
                    WHEN (ce.itemid = ANY (ARRAY[615, 618, 220210, 224690])) AND ce.valuenum > 0::double precision AND ce.valuenum < 70::double precision THEN 5
                    WHEN (ce.itemid = ANY (ARRAY[223761, 678])) AND ce.valuenum > 70::double precision AND ce.valuenum < 120::double precision THEN 6
                    WHEN (ce.itemid = ANY (ARRAY[223762, 676])) AND ce.valuenum > 10::double precision AND ce.valuenum < 50::double precision THEN 6
                    WHEN (ce.itemid = ANY (ARRAY[646, 220277])) AND ce.valuenum > 0::double precision AND ce.valuenum <= 100::double precision THEN 7
                    WHEN (ce.itemid = ANY (ARRAY[442, 455, 220179])) AND ce.valuenum > 0::double precision AND ce.valuenum < 400::double precision THEN 8
                    WHEN (ce.itemid = ANY (ARRAY[8440, 8441, 220180])) AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN 9
                    WHEN (ce.itemid = ANY (ARRAY[456, 443, 220181])) AND ce.valuenum > 0::double precision AND ce.valuenum < 300::double precision THEN 10
                    ELSE NULL::integer
                END AS vitalid,
                CASE
                    WHEN ce.itemid = ANY (ARRAY[223761, 678]) THEN (ce.valuenum - 32::double precision) / 1.8::double precision
                    ELSE ce.valuenum
                END AS valuenum
           FROM mimiciii.icustays ie   -- 选择仅是进入ICU 12~24 小时的相关数据
             LEFT JOIN mimiciii.chartevents ce ON ie.subject_id = ce.subject_id AND ie.hadm_id = ce.hadm_id AND ie.icustay_id = ce.icustay_id AND ce.charttime >= (ie.intime + '12:00:00'::interval hour) AND ce.charttime <= (ie.intime + '1 day'::interval day) AND ce.error IS DISTINCT FROM 1
          WHERE ce.itemid = ANY (ARRAY[211, 220045, 51, 442, 455, 6701, 220179, 220050, 836, 8, 8440, 8441, 8555, 220180, 220051, 456, 52, 6702, 443, 220052, 220181, 225312, 618, 615, 220210, 224690, 646, 220277, 223762, 676, 223761, 678])) pvt
  GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id
  ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id
WITH DATA;

ALTER TABLE mimiciii.new_vitalsfirstday2
  OWNER TO postgres;
