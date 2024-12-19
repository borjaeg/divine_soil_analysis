def write_results(
    writer,
    results_file,
    training_data,
    testing_data,
    min_train_yield,
    max_train_yield,
    min_test_yield,
    max_test_yield,
    num_experiment, 
    num_fold,
    configuration,
    search_space,
    selected_models_type,
    selected_models_weights,
    input_shape,
    best_indices,
    best_indices_2,
    # best_indices_sorted,
    # best_indices_pls,
    # feature_extraction_loss,
    # feature_extraction_val_loss,
    # pre_auto_results,
    # auto_results,
    baseline_results,
    elapsed_time,
):
    configuration_parts = configuration.split("_")
    fill_mode = configuration_parts[0]
    filter_outliers = configuration_parts[1]
    rfe_percent = configuration_parts[2]
    selection_criteria = configuration_parts[3]
    k_splits = configuration_parts[4]
    ensemble_size = configuration_parts[5]
    test_size = configuration_parts[6]
    features_to_select = configuration_parts[7]
    scaling_mode = configuration_parts[8]
    feature_extraction_mode = configuration_parts[9]
    feature_to_extract = configuration_parts[10]
    smoothing_order = configuration_parts[11]
    polynomial_degree = configuration_parts[12]
    mins_search = configuration_parts[13]

    writer.writerow(
        [
            num_experiment,
            num_fold,
            training_data,
            testing_data,
            min_train_yield,
            max_train_yield,
            min_test_yield,
            max_test_yield,
            fill_mode,
            filter_outliers,
            smoothing_order,
            rfe_percent,
            selection_criteria,
            k_splits,
            ensemble_size,
            test_size,
            features_to_select,
            scaling_mode,
            search_space,
            feature_extraction_mode,
            feature_to_extract,
            polynomial_degree,
            mins_search,
            selected_models_type,
            selected_models_weights,
            input_shape,
            best_indices,
            best_indices_2,
            # best_indices_sorted,
            # best_indices_pls,
            # round(feature_extraction_loss, 4),
            # round(feature_extraction_val_loss, 4),
            # *pre_auto_results,
            # *auto_results,
            *baseline_results,
            round(elapsed_time, 4),
        ]
    )
    results_file.flush()
