SELECT
	CONCAT(features_to_select, '-', selection_criteria, '-', feature_extractor, '-', extracted_features) pipeline,
	ROUND(MAX(score), 4) score
FROM
	(
		SELECT
			*
		FROM
			(
				SELECT
					-- train_data,
					fill_mode,
					outliers_filtered,
					smoothing_window,
					use_rfe,
					scaling_mode,
					features_to_select,
					feature_extractor,
					extracted_features,
					selection_criteria,
					-- ROUND(AVG(ard_test_r2_score), 5) score,
					-- ROUND(AVG(pls_test_r2_score), 5) score,
					ROUND(AVG(mlp_test_r2_score), 5) score,
					COUNT(*) num
				FROM
					divine_soil_all_depths_cv_paper_4
				WHERE
					-- ABS(ard_train_r2_score - ard_test_r2_score) < 0.25
					-- AND ard_test_r2_score > 0.
					-- AND ard_adjusted_test_r2_score > 0.
					-- AND ard_adjusted_test_r2_score < 1.0
					-- ABS(pls_train_r2_score - pls_test_r2_score) < 0.25
					-- AND pls_test_r2_score > 0.0
					-- AND pls_adjusted_test_r2_score > 0
					-- AND pls_adjusted_test_r2_score < 1
					ABS(mlp_train_r2_score - mlp_test_r2_score) < 0.2
					AND mlp_test_r2_score > 0.0
					AND mlp_adjusted_test_r2_score > 0
					AND mlp_adjusted_test_r2_score < 1
					AND train_data IN ('Mg')
				GROUP BY
					-- train_data,
					fill_mode,
					outliers_filtered,
					smoothing_window,
					use_rfe,
					scaling_mode,
					features_to_select,
					feature_extractor,
					extracted_features,
					selection_criteria
			) T_filtering_good
		WHERE
			num >= 8 -- 60
	) T_report
GROUP BY
	features_to_select,
	selection_criteria,
	feature_extractor,
	extracted_features
ORDER BY
	score DESC;


SELECT
	train_data,
	selection_criteria,
	features_to_select,
	feature_extractor,
	extracted_features,
	MAX(test_r2_score) test_r2_score
FROM
	(
		SELECT
			train_data,
			use_rfe,
			selection_criteria,
			features_to_select,
			feature_extractor,
			extracted_features,
			-- ROUND(MAX(ard_test_r2_score), 4) test_r2_score
			-- ROUND(MAX(pls_test_r2_score), 4) test_r2_score
			ROUND(MAX(mlp_test_r2_score), 4) test_r2_score
		FROM
			(
				SELECT
					*
				FROM
					(
						SELECT
							train_data,
							fill_mode,
							-- outliers_filtered,
							smoothing_window,
							use_rfe,
							selection_criteria,
							features_to_select,
							scaling_mode,
							feature_extractor,
							extracted_features,
							ROUND(AVG(ard_test_r2_score), 4) ard_test_r2_score,
							ROUND(AVG(pls_test_r2_score), 4) pls_test_r2_score,
							ROUND(AVG(mlp_test_r2_score), 4) mlp_test_r2_score,
							COUNT(*) num
						FROM
							divine_soil_all_depths_cv_paper_4
						 WHERE
							-- ABS(ard_train_r2_score - ard_test_r2_score) < 0.2
							-- AND ard_test_r2_score > 0
							-- AND ard_adjusted_test_r2_score > 0.0
							-- AND ard_adjusted_test_r2_score < 1.0
							
							-- ABS(pls_train_r2_score - pls_test_r2_score) < 0.25
						    -- AND pls_test_r2_score > 0.0
						 	-- AND pls_adjusted_test_r2_score > 0
							-- AND pls_adjusted_test_r2_score < 1
							
							ABS(mlp_train_r2_score - mlp_test_r2_score) < 0.2
							AND mlp_test_r2_score > 0.0
							AND mlp_adjusted_test_r2_score > 0
							AND mlp_adjusted_test_r2_score < 1
						GROUP BY
							train_data,
							fill_mode,
							-- outliers_filtered,
							smoothing_window,
							use_rfe,
							selection_criteria,
							features_to_select,
							scaling_mode,
							feature_extractor,
							extracted_features
					) T
				WHERE
					num >= 8
			) T_min_results
		GROUP BY
			train_data,
			use_rfe,
			features_to_select,
			selection_criteria,
			feature_extractor,
			extracted_features
		ORDER BY
			train_data ASC,
			test_r2_score DESC
	) T_filter_fields
GROUP BY
	train_data,
	selection_criteria,
	features_to_select,
	feature_extractor,
	extracted_features
;

CREATE TABLE `divine_soil_all_depths_cv_paper_4` (
  `num_experiment` int DEFAULT NULL,
  `num_fold` int DEFAULT NULL,
  `train_data` varchar(100) DEFAULT NULL,
  `test_data` varchar(20) DEFAULT NULL,
  `min_train_data` float DEFAULT NULL,
  `max_train_data` float DEFAULT NULL,
  `min_test_data` float DEFAULT NULL,
  `max_test_data` float DEFAULT NULL,
  `fill_mode` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `outliers_filtered` int DEFAULT NULL,
  `smoothing_window` varchar(5) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `use_rfe` varchar(5) DEFAULT NULL,
  `selection_criteria` varchar(30) DEFAULT NULL,
  `k_splits` float DEFAULT NULL,
  `ensemble_size` int DEFAULT NULL,
  `test_size` int DEFAULT NULL,
  `features_to_select` varchar(10) DEFAULT NULL,
  `scaling_mode` varchar(10) DEFAULT NULL,
  `search_space` int DEFAULT NULL,
  `feature_extractor` varchar(15) DEFAULT NULL,
  `extracted_features` varchar(10) DEFAULT NULL,
  `polynomial_degree` int DEFAULT NULL,
  `mins_training` int DEFAULT NULL,
  `best_models` varchar(200) DEFAULT NULL,
  `ensemble_weights` varchar(200) DEFAULT NULL,
  `input_shape` varchar(20) DEFAULT NULL,
  `best_features_1` text,
  `best_features_2` text,
  `ard_train_r2_score` double DEFAULT NULL,
  `ard_test_r2_score` double DEFAULT NULL,
  `ard_adjusted_train_r2_score` double DEFAULT NULL,
  `ard_adjusted_test_r2_score` double DEFAULT NULL,
  `ard_train_rmse_score` double DEFAULT NULL,
  `ard_test_rmse_score` double DEFAULT NULL,
  `ard_coefficients` text,
  `mlp_train_r2_score` double DEFAULT NULL,
  `mlp_test_r2_score` double DEFAULT NULL,
  `mlp_adjusted_train_r2_score` double DEFAULT NULL,
  `mlp_adjusted_test_r2_score` double DEFAULT NULL,
  `mlp_train_rmse_score` double DEFAULT NULL,
  `mlp_test_rmse_score` double DEFAULT NULL,
  `poly_ard_train_r2_score` double DEFAULT NULL,
  `poly_ard_test_r2_score` double DEFAULT NULL,
  `poly_ard_adjusted_train_r2_score` double DEFAULT NULL,
  `poly_ard_adjusted_test_r2_score` double DEFAULT NULL,
  `poly_ard_train_rmse_score` double DEFAULT NULL,
  `poly_ard_test_rmse_score` double DEFAULT NULL,
  `trans_poly_ard_train_r2_score` double DEFAULT NULL,
  `trans_poly_ard_test_r2_score` double DEFAULT NULL,
  `trans_poly_ard_adjusted_train_r2_score` double DEFAULT NULL,
  `trans_poly_ard_adjusted_test_r2_score` double DEFAULT NULL,
  `trans_poly_ard_train_rmse_score` double DEFAULT NULL,
  `trans_poly_ard_test_rmse_score` double DEFAULT NULL,
  `trans_ard_train_r2_score` double DEFAULT NULL,
  `trans_ard_test_r2_score` double DEFAULT NULL,
  `trans_ard_adjusted_train_r2_score` double DEFAULT NULL,
  `trans_ard_adjusted_test_r2_score` double DEFAULT NULL,
  `trans_ard_train_rmse_score` double DEFAULT NULL,
  `trans_ard_test_rmse_score` double DEFAULT NULL,
  `ridge_train_r2_score` double DEFAULT NULL,
  `ridge_test_r2_score` double DEFAULT NULL,
  `ridge_adjusted_train_r2_score` double DEFAULT NULL,
  `ridge_adjusted_test_r2_score` double DEFAULT NULL,
  `ridge_train_rmse_score` double DEFAULT NULL,
  `ridge_test_rmse_score` double DEFAULT NULL,
  `poly_svr_train_r2_score` double DEFAULT NULL,
  `poly_svr_test_r2_score` double DEFAULT NULL,
  `poly_svr_adjusted_train_r2_score` double DEFAULT NULL,
  `poly_svr_adjusted_test_r2_score` double DEFAULT NULL,
  `poly_svr_train_rmse_score` double DEFAULT NULL,
  `poly_svr_test_rmse_score` double DEFAULT NULL,
  `poly_svr_coefficients` text,
  `bayesian_train_r2_score` double DEFAULT NULL,
  `bayesian_test_r2_score` double DEFAULT NULL,
  `bayesian_adjusted_train_r2_score` double DEFAULT NULL,
  `bayesian_adjusted_test_r2_score` double DEFAULT NULL,
  `bayesian_train_rmse_score` double DEFAULT NULL,
  `bayesian_test_rmse_score` double DEFAULT NULL,
  `pls_train_r2_score` double DEFAULT NULL,
  `pls_test_r2_score` double DEFAULT NULL,
  `pls_adjusted_train_r2_score` double DEFAULT NULL,
  `pls_adjusted_test_r2_score` double DEFAULT NULL,
  `pls_train_rmse_score` double DEFAULT NULL,
  `pls_test_rmse_score` double DEFAULT NULL,
  `elapsed_time` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;