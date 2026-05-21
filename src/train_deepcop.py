from deepcop_target.train_deepcop_target import main


if __name__ == "__main__":
    main()
# warm: train_n=42576 test_n=10644 | test_MSE=0.8299 | test_PCC=0.5422
# cold_target_pattern: train_n=45678 test_n=7542 | test_MSE=0.9852 | test_PCC=0.5218
# cold_cell: train_n=48873 test_n=4347 | test_MSE=1.2677 | test_PCC=0.3223


# ===== GSNN Summary =====
# warm: train_n=42576 test_n=10644 | test_MSE=0.9341 | test_PCC=0.4609
# cold_target_pattern: train_n=45678 test_n=7542 | test_MSE=1.0840 | test_PCC=0.4593
# cold_cell: train_n=48873 test_n=4347 | test_MSE=0.8503 | test_PCC=0.4218


# warm: train_n=42576 test_n=10644 | test_MSE=0.7448 | test_PCC=0.5229
# cold_target_pattern: train_n=45678 test_n=7542 | test_MSE=1.0546 | test_PCC=0.3886
# cold_cell: train_n=48873 test_n=4347 | test_MSE=0.8490 | test_PCC=0.3014


# warm: train_n=42576 test_n=10644 | test_MSE=0.7438 | test_PCC=0.5239
# cold_target_pattern: train_n=45678 test_n=7542 | test_MSE=1.0432 | test_PCC=0.3981
# cold_cell: train_n=48873 test_n=4347 | test_MSE=0.8516 | test_PCC=0.3027


# ===== Summary =====
# warm: train_n=42576 test_n=10644 | test_MSE=0.8835 | test_PCC=0.4897
# cold_target_pattern: train_n=45678 test_n=7542 | test_MSE=1.0468 | test_PCC=0.4682
# cold_cell: train_n=48873 test_n=4347 | test_MSE=0.8415 | test_PCC=0.4056
