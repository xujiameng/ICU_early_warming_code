-- Materialized View: mimiciii.new_ready_h12_runin

-- DROP MATERIALIZED VIEW mimiciii.new_ready_h12_runin;

-- Description
--   该程序功能为：提取患者进入ICU后12-24小时无创血压数据及有创血压数据
--   程序流程：根据官网提供的代码获取血压数据，并对血压值的异常数据进行处理

--   程序结果：生成患者进入ICU后12-24小时的血压数据表



CREATE MATERIALIZED VIEW mimiciii.new_ready_h12_runin AS 
 WITH chart_value AS (
         SELECT ie.subject_id,
            ie.hadm_id,
            ie.icustay_id,
                CASE
                    WHEN le.itemid = ANY (ARRAY[51, 6701, 220050]) THEN 'sysbp'::text			--根据官网提供的代码获取有创及无创数据
                    WHEN le.itemid = ANY (ARRAY[8368, 8555, 220051]) THEN 'diasbp'::text
                    WHEN le.itemid = ANY (ARRAY[52, 6702, 220052, 225312]) THEN 'meanbp'::text
                    WHEN le.itemid = ANY (ARRAY[442, 455, 220179]) THEN 'nisysbp'::text
                    WHEN le.itemid = ANY (ARRAY[8440, 8441, 220180]) THEN 'nidiasbp'::text
                    WHEN le.itemid = ANY (ARRAY[456, 443, 220181]) THEN 'nimeanbp'::text
                    ELSE NULL::text
                END AS label,
            le.charttime,
            le.itemid,
                CASE
                    WHEN (le.itemid = ANY (ARRAY[51, 6701, 220050])) AND le.valuenum > 0::double precision AND le.valuenum < 400::double precision THEN le.valuenum		--处理异常值
                    WHEN (le.itemid = ANY (ARRAY[8368, 8555, 220051])) AND le.valuenum > 0::double precision AND le.valuenum < 300::double precision THEN le.valuenum
                    WHEN (le.itemid = ANY (ARRAY[52, 6702, 220052, 225312])) AND le.valuenum > 0::double precision AND le.valuenum < 300::double precision THEN le.valuenum
                    WHEN (le.itemid = ANY (ARRAY[442, 455, 220179])) AND le.valuenum > 0::double precision AND le.valuenum < 400::double precision THEN le.valuenum
                    WHEN (le.itemid = ANY (ARRAY[8440, 8441, 220180])) AND le.valuenum > 0::double precision AND le.valuenum < 300::double precision THEN le.valuenum
                    WHEN (le.itemid = ANY (ARRAY[456, 443, 220181])) AND le.valuenum > 0::double precision AND le.valuenum < 300::double precision THEN le.valuenum
                    ELSE NULL::double precision
                END AS valuenum
           FROM mimiciii.icustays ie
             LEFT JOIN mimiciii.chartevents le ON le.subject_id = ie.subject_id AND le.hadm_id = ie.hadm_id AND ie.icustay_id = le.icustay_id AND le.charttime >= ie.intime AND le.charttime <= ie.outtime AND le.charttime >= (ie.intime + '12:00:00'::interval hour) AND le.charttime <= (ie.intime + '1 day'::interval day) AND le.error IS DISTINCT FROM 1 AND (le.itemid = ANY (ARRAY[51, 6701, 220050, 8368, 8555, 220051, 52, 6702, 220052, 225312, 442, 455, 220179, 8440, 8441, 220180, 456, 443, 220181]))     --提取患者进入ICU12-24小时信息的条件
        )
 SELECT pvt.subject_id,
    pvt.hadm_id,
    pvt.icustay_id,
    pvt.charttime,
    max(				--此处获取最大值是为了避免出现同一患者，同一时间出现多个数据的异常情况
        CASE
            WHEN pvt.label = 'sysbp'::text THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp,
    max(
        CASE
            WHEN pvt.label = 'diasbp'::text THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp,
    max(
        CASE
            WHEN pvt.label = 'meanbp'::text THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp,
    max(
        CASE
            WHEN pvt.label = 'nisysbp'::text THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nisysbp,
    max(
        CASE
            WHEN pvt.label = 'nidiasbp'::text THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nidiasbp,
    max(
        CASE
            WHEN pvt.label = 'nimeanbp'::text THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS nimeanbp
   FROM chart_value pvt
  GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime
  ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime
WITH DATA;

ALTER TABLE mimiciii.new_ready_h12_runin
  OWNER TO postgres;
