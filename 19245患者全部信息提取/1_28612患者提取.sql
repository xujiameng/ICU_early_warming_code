-- Materialized View: mimiciii.select_pat_without_label

-- DROP MATERIALIZED VIEW mimiciii.select_pat_without_label;


-- Description
--   该程序功能为：提取满足以下条件的患者，即以下患者满足模型要求，但尚未进行数据预处理
--   程序流程为：
--	Step1：提取仅进入过一次ICU的病例
--	Step2：提取仅进入过一次医院的病例
--	Step3：提取年龄大于等于16岁的病例
--	Step4：剔除进入ICU后30天内院外死亡的病例，且将30天后死亡病例视为存活

--   程序运行结果为：生成满足条件的28612患者初始信息的表

--  V1.0:2018/8/28


CREATE MATERIALIZED VIEW mimiciii.select_pat_without_label AS 
 WITH co AS (                                                    --获取MIMICIII数据库患者基本信息
         SELECT icustays.subject_id,
            icustays.hadm_id,
            icustays.icustay_id,
            icustays.first_careunit,
            icustays.intime,
            icustays.outtime,
            icustays.los
           FROM mimiciii.icustays
        ), linshit AS (                                           --获取患者进入ICU的次数
         SELECT co.subject_id,
            co.hadm_id,
            co.icustay_id,
            co.first_careunit,
            co.intime,
            co.outtime,
            co.los,
            count(*) OVER (PARTITION BY co.subject_id ORDER BY co.subject_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first
           FROM co
        ), oneicu AS (                                           --获取仅进入过一次ICU的患者的信息
         SELECT linshit.subject_id,
            linshit.hadm_id,
            linshit.icustay_id,
            linshit.intime,
            linshit.outtime,
            linshit.los,
                CASE
                    WHEN linshit.first = 1 THEN 1
                    ELSE 0
                END AS inclusion_icu
           FROM linshit
          ORDER BY linshit.subject_id
        ), co1 AS (                                           --整合信息，为后续操作做准备
         SELECT admissions.subject_id,
            admissions.hadm_id,
            admissions.deathtime,
            admissions.hospital_expire_flag,
            oneicu.intime,
            oneicu.inclusion_icu
           FROM oneicu
             LEFT JOIN mimiciii.admissions ON admissions.hadm_id = oneicu.hadm_id
        ), linshittper AS (                                          --整合信息，为后续操作做准备
         SELECT admissions.row_id,
            admissions.subject_id,
            admissions.hadm_id,
            admissions.admittime,
            admissions.dischtime,
            admissions.deathtime,
            admissions.admission_type,
            admissions.admission_location,
            admissions.discharge_location,
            admissions.insurance,
            admissions.language,
            admissions.religion,
            admissions.marital_status,
            admissions.ethnicity,
            admissions.edregtime,
            admissions.edouttime,
            admissions.diagnosis,
            admissions.hospital_expire_flag,
            admissions.has_chartevents_data,
            count(*) OVER (PARTITION BY admissions.subject_id ORDER BY admissions.subject_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS second -- 获取患者进入医院的次数
           FROM mimiciii.admissions
        ), linshitt AS ( 				  --转换 second 变量
         SELECT linshittper.subject_id,
            linshittper.hadm_id,
            linshittper.deathtime,
            linshittper.hospital_expire_flag,
            co1.intime,
            co1.inclusion_icu,
            linshittper.second
           FROM co1
             LEFT JOIN linshittper ON linshittper.hadm_id = co1.hadm_id
        ), onehospital AS (				-- 获取仅进去过一次医院的患者信息
         SELECT linshitt.subject_id,
            linshitt.hadm_id,
            linshitt.deathtime,
            linshitt.hospital_expire_flag,
            linshitt.intime,
            linshitt.inclusion_icu,
                CASE
                    WHEN linshitt.second = 1 THEN 1
                    ELSE 0
                END AS inclusion_hos
           FROM linshitt
          ORDER BY linshitt.subject_id
        ), co2per AS (				-- 计算患者年龄
         SELECT icupat.subject_id,
            icupat.dob,
            icupat.intime,
            icupat.age,
            icupat.hadm_id
           FROM ( SELECT icustays.hadm_id,
                    patients.subject_id,
                    patients.dob,
                    icustays.intime,
                    date_part('year'::text, icustays.intime) - date_part('year'::text, patients.dob) AS age
                   FROM mimiciii.icustays
                     LEFT JOIN mimiciii.patients ON patients.subject_id = icustays.subject_id) icupat
        ), co2 AS (			-- 转换 age 变量
         SELECT co2per.subject_id,
            co2per.dob,
            co2per.intime,
            co2per.age,
            co2per.hadm_id,
            onehospital.inclusion_icu,
            onehospital.inclusion_hos
           FROM onehospital
             LEFT JOIN co2per ON co2per.subject_id = onehospital.subject_id AND co2per.hadm_id = onehospital.hadm_id AND co2per.intime = onehospital.intime
        ), age AS (    			--获取年龄大于15的患者的信息
         SELECT co2.subject_id,
            co2.age,
            co2.hadm_id,
            co2.intime,
            onehospital.inclusion_icu,
            onehospital.inclusion_hos,
                CASE
                    WHEN co2.age > 15::double precision THEN 1
                    ELSE 0
                END AS inclusion_age
           FROM co2
             LEFT JOIN onehospital ON co2.subject_id = onehospital.subject_id AND co2.hadm_id = onehospital.hadm_id AND co2.intime = onehospital.intime
          ORDER BY co2.subject_id
        ), linshi2 AS (				--数据整合
         SELECT oncepat.subject_id,
            oncepat.expire_flag,
            oncepat.hospital_expire_flag,
            oncepat.dod,
            oncepat.hadm_id,
            oncepat.intime
           FROM ( SELECT pat.subject_id,
                    pat.expire_flag,
                    one.hospital_expire_flag,
                    pat.dod,
                    one.hadm_id,
                    one.intime
                   FROM onehospital one
                     LEFT JOIN mimiciii.patients pat ON pat.subject_id = one.subject_id) oncepat
          ORDER BY oncepat.subject_id
        ), inhospital AS (			--住院信息提取
         SELECT linshi2.subject_id,
            linshi2.expire_flag,
            linshi2.hospital_expire_flag,
            linshi2.dod,
            icu.intime,
            icu.hadm_id
           FROM mimiciii.icustays icu
             LEFT JOIN linshi2 ON linshi2.subject_id = icu.subject_id AND linshi2.hadm_id = icu.hadm_id AND linshi2.intime = icu.intime
        ), deathlabel AS (   			-- 准备剔除进入ICU后30天内在院外死亡的病例
         SELECT bg.subject_id,
            bg.expire_flag,
            bg.hospital_expire_flag,
            bg.dod,
            bg.intime,
            bg.*::record AS bg,
            bg.hadm_id,
                CASE
                    WHEN bg.expire_flag = 0 THEN 1            --对于没有死亡的患者纳入考虑
                    WHEN bg.expire_flag = 1 THEN	--对于死亡的患者进行如下判断
                    CASE
                        WHEN bg.hospital_expire_flag = 0 THEN
                        CASE
                            WHEN bg.dod < (bg.intime + '30 days'::interval day) THEN 0     --院外入ICU小于30天不满足条件
                            WHEN bg.dod >= (bg.intime + '30 days'::interval day) THEN 1    --院外入ICU大于30天等待后续处理
                            ELSE NULL::integer
                        END
                        WHEN bg.hospital_expire_flag = 1 THEN
                        CASE
                            WHEN bg.dod < (bg.intime + '30 days'::interval day) THEN 1    --院内入ICU小于30天等待后续处理
                            WHEN bg.dod >= (bg.intime + '30 days'::interval day) THEN 1    --院内入ICU大于30天等待后续处理
                            ELSE NULL::integer
                        END
                        ELSE NULL::integer
                    END
                    ELSE NULL::integer
                END AS inclusion_died_in_30_hos
           FROM inhospital bg
        ), deathlabel2 AS (
         SELECT bg.subject_id,
            bg.expire_flag,
            bg.hospital_expire_flag,
            bg.dod,
            bg.intime,
            bg.*::record AS bg,
            bg.hadm_id,
                CASE
                    WHEN bg.expire_flag = 0 THEN 0          -- 将存活的患者视为存活
                    WHEN bg.expire_flag = 1 THEN             -- 对最终死亡的患者进行如下判断
                    CASE
                        WHEN bg.hospital_expire_flag = 0 THEN
                        CASE
                            WHEN bg.dod < (bg.intime + '30 days'::interval day) THEN 2    --剔除进入ICU后30天内在院外死亡的病例
                            WHEN bg.dod >= (bg.intime + '30 days'::interval day) THEN 0   --30天后院外死亡视为存活
                            ELSE NULL::integer
                        END
                        WHEN bg.hospital_expire_flag = 1 THEN
                        CASE
                            WHEN bg.dod < (bg.intime + '30 days'::interval day) THEN 1   --30天内院内死亡视为死亡
                            WHEN bg.dod >= (bg.intime + '30 days'::interval day) THEN 0    --30天后院内死亡视为存活
                            ELSE NULL::integer
                        END
                        ELSE NULL::integer
                    END
                    ELSE NULL::integer
                END AS death
           FROM inhospital bg
        ), linshi1 AS (				--数据整合
         SELECT icupat.subject_id,
            icupat.gender,
            icupat.first_careunit,
            icupat.age,
            icupat.intime,
            icupat.hadm_id
           FROM ( SELECT patients.subject_id,
                    patients.gender,
                    icustays.first_careunit,
                    icustays.intime,
                    icustays.hadm_id,
                    date_part('year'::text, icustays.intime) - date_part('year'::text, patients.dob) AS age
                   FROM mimiciii.icustays
                     LEFT JOIN mimiciii.patients ON patients.subject_id = icustays.subject_id) icupat
        ), baseinfo1 AS (			--标记年龄大于15岁的患者
         SELECT linshi1.subject_id,
            linshi1.age,
            linshi1.gender,
            linshi1.first_careunit,
            linshi1.intime,
            linshi1.hadm_id,
                CASE
                    WHEN linshi1.age > 15::double precision THEN 1
                    ELSE 0
                END AS agelabel
           FROM linshi1
          ORDER BY linshi1.subject_id
        ), co4 AS (			 		--根据官网提供的对应编码，获取患者其它基本信息
         SELECT oneicu_base.subject_id,
            oneicu_base.intime,
            oneicu_base.hadm_id,
            oneicu_base.icustay_id,
            oneicu_base.age,
            oneicu_base.gender,
            oneicu_base.first_careunit
           FROM ( SELECT oneicu.subject_id,
                    oneicu.hadm_id,
                    oneicu.icustay_id,
                    oneicu.intime,
                        CASE
                            WHEN base.age = 300::double precision THEN 91::double precision
                            ELSE base.age
                        END AS age,
                        CASE
                            WHEN base.gender::text = 'M'::text THEN 1
                            ELSE 0
                        END AS gender,
                        CASE
                            WHEN base.first_careunit::text = 'CCU'::text THEN 1
                            WHEN base.first_careunit::text = 'CSRU'::text THEN 2
                            WHEN base.first_careunit::text = 'MICU'::text THEN 3
                            WHEN base.first_careunit::text = 'SICU'::text THEN 4
                            WHEN base.first_careunit::text = 'TSICU'::text THEN 5
                            ELSE 0
                        END AS first_careunit
                   FROM oneicu
                     LEFT JOIN baseinfo1 base ON oneicu.subject_id = base.subject_id AND oneicu.intime = base.intime AND oneicu.hadm_id = base.hadm_id) oneicu_base
          WHERE (oneicu_base.subject_id IN ( SELECT age.subject_id
                   FROM age))
        ), select_pat AS (			--数据整合
         SELECT co4.subject_id,
            co4.hadm_id,
            co4.intime,
            co4.icustay_id,
            co4.age,
            co4.gender,
            co4.first_careunit,
            deathlabel.inclusion_died_in_30_hos
           FROM deathlabel
             LEFT JOIN co4 ON co4.subject_id = deathlabel.subject_id AND co4.hadm_id = deathlabel.hadm_id AND co4.intime = deathlabel.intime AND (deathlabel.inclusion_died_in_30_hos = 0 OR deathlabel.inclusion_died_in_30_hos = 1)
        ), the_last_step_in_pat AS ( 			--根据年龄进行选择标记
         SELECT select_pat.subject_id,
            select_pat.hadm_id,
            select_pat.intime,
            select_pat.age,
            select_pat.icustay_id,
            age.inclusion_icu,
            age.inclusion_hos,
            age.inclusion_age,
            select_pat.inclusion_died_in_30_hos,
                CASE
                    WHEN age.inclusion_icu = 1 AND age.inclusion_hos = 1 AND age.inclusion_age = 1 AND select_pat.inclusion_died_in_30_hos = 1 THEN 1
                    ELSE 0
                END AS selected
           FROM age
             LEFT JOIN select_pat ON age.subject_id = select_pat.subject_id AND age.hadm_id = select_pat.hadm_id AND age.intime = select_pat.intime
        )
 SELECT the_last_step_in_pat.subject_id,   			--信息整合
    the_last_step_in_pat.hadm_id,
    the_last_step_in_pat.intime,
    the_last_step_in_pat.age,
    the_last_step_in_pat.icustay_id,
    deathlabel2.death
   FROM the_last_step_in_pat
     LEFT JOIN deathlabel2 ON the_last_step_in_pat.subject_id = deathlabel2.subject_id AND the_last_step_in_pat.hadm_id = deathlabel2.hadm_id AND the_last_step_in_pat.intime = deathlabel2.intime
  WHERE the_last_step_in_pat.selected = 1
WITH DATA;

ALTER TABLE mimiciii.select_pat_without_label
  OWNER TO postgres;
