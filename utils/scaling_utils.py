from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_input_data(
    train_ndvi_filtered,
    test_ndvi_filtered,
    scaling_mode: "str",
    use_holdout_scaling: bool,
):
    if use_holdout_scaling:
        if scaling_mode == "minmax":
            t = MinMaxScaler()
            train_ndvi_input_scaled = t.fit_transform(train_ndvi_filtered)
            test_ndvi_input_scaled = t.fit_transform(test_ndvi_filtered)
        elif scaling_mode == "standard":
            t = StandardScaler()
            train_ndvi_input_scaled = t.fit_transform(train_ndvi_filtered)
            test_ndvi_input_scaled = t.fit_transform(test_ndvi_filtered)
        else:
            train_ndvi_input_scaled = train_ndvi_filtered
            test_ndvi_input_scaled = test_ndvi_filtered
    else:
        if scaling_mode == "minmax":
            t = MinMaxScaler()
            train_ndvi_input_scaled = t.fit_transform(train_ndvi_filtered)
            test_ndvi_input_scaled = t.transform(test_ndvi_filtered)
        elif scaling_mode == "standard":
            t = StandardScaler()
            train_ndvi_input_scaled = t.fit_transform(train_ndvi_filtered)
            test_ndvi_input_scaled = t.transform(test_ndvi_filtered)
        else:
            train_ndvi_input_scaled = train_ndvi_filtered
            test_ndvi_input_scaled = test_ndvi_filtered

    return train_ndvi_input_scaled, test_ndvi_input_scaled
