-- Materialized View: mimiciii.new_vari1

-- DROP MATERIALIZED VIEW mimiciii.new_vari1;


-- Description 
--   该程序功能为：计算患者进入ICU第一天人口统计变量等数据的衍生变量
--   程序流程为：选择符合要求的患者生理数据，包括有患者编号，ICU类型，心率，呼吸频率，体温，血氧饱和度，无创收缩压、舒张压、平均压
--					     年龄，性别，是否死亡，是否进行机械通气，总尿量，BMI，昏迷指数，吸入氧浓度
--   程序运行结果；生成一个包含患者进入ICU 第一天的人口统计变量等数据的衍生变量的表



CREATE MATERIALIZED VIEW mimiciii.new_vari1 AS 
 WITH co AS (
         SELECT a_1.subject_id,
            a_1.age,
            a_1.gender,
            a_1.first_careunit AS icutype,
            a_1.death,
            a_1.urineoutput,
            a_1.bmi,
            b_1.mingcs AS qgcs,
            c_1.fio2_min AS qfio2min,
            c_1.fio2_max AS qfio2max,
            c_1.fio2_mean AS qfio2mean,
            c_1.fio2_var AS qfio2var,
            c_1.fio2_stddev AS qfio2stddev,
            c_1.range AS qfio2range,
            c_1.perc_25 AS qfio2perc_25,
            c_1.perc_50 AS qfio2perc_50,
            c_1.perc_75 AS qfio2perc_75
           FROM mimiciii.finalbase a_1
             LEFT JOIN mimiciii.gcsfirstday1 b_1 ON a_1.subject_id = b_1.subject_id AND a_1.hadm_id = b_1.hadm_id AND a_1.icustay_id = b_1.icustay_id
             LEFT JOIN mimiciii.new_vitalsfio21 c_1 ON a_1.subject_id = c_1.subject_id AND a_1.hadm_id = c_1.hadm_id AND a_1.icustay_id = c_1.icustay_id
        )
 SELECT a.subject_id,
    a.age,
    a.gender,
    a.icutype,
    a.death,
    a.urineoutput,
    a.bmi,
    a.qgcs,
    b.mingcs AS hgcs,
    a.qfio2min,
    a.qfio2max,
    a.qfio2mean,
    a.qfio2var,
    a.qfio2stddev,
    a.qfio2range,
    a.qfio2perc_25,
    a.qfio2perc_50,
    a.qfio2perc_75,
    c.fio2_min AS hfio2min,
    c.fio2_max AS hfio2max,
    c.fio2_mean AS hfio2mean,
    c.fio2_var AS hfio2var,
    c.fio2_stddev AS hfio2stddev,
    c.range AS hfio2range,
    c.perc_25 AS hfio2perc_25,
    c.perc_50 AS hfio2perc_50,
    c.perc_75 AS hfio2perc_75
   FROM co a
     LEFT JOIN mimiciii.gcsfirstday2 b ON a.subject_id = b.subject_id
     LEFT JOIN mimiciii.new_vitalsfio22 c ON a.subject_id = c.subject_id
  ORDER BY a.subject_id
WITH DATA;

ALTER TABLE mimiciii.new_vari1
  OWNER TO postgres;
