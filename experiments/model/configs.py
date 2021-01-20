import cv2

PKG_PATH = 'gelsight_simulation'

# {'position': [0, 1, 0.25], 'color': (240, 240, 240)},
# {'position': [-1, 0, 0.25], 'color': (255, 139, 78)},
# {'position': [0, -1, 0.25], 'color': (108, 82, 255)},
# {'position': [1, 0, 0.25], 'color': (100, 240, 150)},
smartlab_gelsight2014_config = {
    'light_sources': [
        {'position': [0, 1, 0.25], 'color': (255, 255, 255), 'kd': 0.6, 'ks': 0.5},  # white, top
        {'position': [-1, 0, 0.25], 'color': (255, 130, 115), 'kd': 0.5, 'ks': 0.3},  # blue, right
        {'position': [0, -1, 0.25], 'color': (108, 82, 255), 'kd': 0.6, 'ks': 0.4},  # red, bottom
        {'position': [1, 0, 0.25], 'color': (120, 255, 153), 'kd': 0.1, 'ks': 0.1}  # green, left

        # {'position': [-1, 0, 0.25], 'color': (255, 255, 255), 'kd': 0.5, 'ks': 0.3},  # white, top
        # {'position': [0, 0, 1], 'color': (255, 255, 255), 'kd': 0.5, 'ks': 0.3},  # blue, right

    ],
    'background_img': cv2.imread(PKG_PATH + '/experiments/fine_tuning/background.png'),
    'ka': 0.8,
    'px2m_ratio': 5.4347826087e-05,
    'elastomer_thickness': 0.004,
    'min_depth': 0.026,
    'texture_sigma': 0.000002
}

mit_gelsight2017_config = {
    'light_sources': [
        {'position': [-1, 0, 0.25], 'color': (108, 82, 255), 'kd': 0.6, 'ks': 0.4},  # red, bottom
        {'position': [0.50, -0.866, 0.25], 'color': (120, 255, 153), 'kd': 0.1, 'ks': 0.4},  # green, left
        {'position': [0.50, 0.866, 0.25], 'color': (255, 130, 115), 'kd': 0.1, 'ks': 0.4},  # blue, left
    ],
    'background_img': cv2.imread(PKG_PATH + '/experiments/fine_tuning/background_gelsight2017.jpg'),
    'ka': 0.8,
    'px2m_ratio': 5.4347826087e-05,
    'elastomer_thickness': 0.004,
    'min_depth': 0.026,
    'texture_sigma': 0.000002
}
