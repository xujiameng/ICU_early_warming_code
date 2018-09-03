-- Materialized View: mimiciii.new_runin

-- DROP MATERIALIZED VIEW mimiciii.new_runin;

-- Description
--   该程序功能为：整合血压与其它变量，为接下来的拟合做准备
--   程序运行结果：生成一个包含有血压及其充当为插补变量的参数的表





CREATE MATERIALIZED VIEW mimiciii.new_runin AS 
 SELECT n.subject_id,
    n.hadm_id,
    n.icustay_id,
    n.charttime,
    nc.age,
    nc.bmi,
    nc.gender,
    nc.icutype,
    nc.death,
    nc.vent,
    n.sysbp,
    n.diasbp,
    n.meanbp,
    n.nisysbp,
    n.nidiasbp,
    n.nimeanbp
   FROM mimiciii.new_ready_runin n,
    mimiciii.new_pat_with_characteristic_notcat nc
  WHERE n.subject_id = nc.subject_id
  ORDER BY n.subject_id, n.hadm_id, n.icustay_id, n.charttime
WITH DATA;

ALTER TABLE mimiciii.new_runin
  OWNER TO postgres;
