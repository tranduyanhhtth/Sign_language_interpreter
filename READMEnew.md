[Webcam] --> [create_gestures.py_(Chup_anh_&_luu_cu_chi)] --> [ gestures/*/*.jpg ] --> [Rotate_images.py_(Lam_giau)] --> [ jpg (tang gap doi) ]
|
v
[load_images.py_(Tao_tap_du_lieu)] --> [ train_images, train_labels, test_images, test_labels, val_images, val_labels ]
|
v
[cnn_model_train.py_(Huan_luyen_mo_hinh_CNN)] --> [ cnn_model_keras2.h5 ]
|
v
[final.py_(Nhan_dien_cu_chi_thoi_gian_thuc)] <--> [ cnn_model_keras2.h5, gestures.json ]
