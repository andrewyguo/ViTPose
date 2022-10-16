dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(

    ),
    keypoint_info={
        0:
        dict(
            name='LUF',
            id=0,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        1:
        dict(
            name='RUF',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='RDF',
            id=2,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        3:
        dict(
            name='LDF',
            id=3,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        4:
        dict(
            name='LUB',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        5:
        dict(
            name='RUB',
            id=5,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        6:
        dict(
            name='RDB',
            id=6,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        7:
        dict(
            name='LDB',
            id=7,
            color=[255, 128, 0],
            type='lower',
            swap=''),
        8:
        dict(
            name='CENTER',
            id=8,
            color=[255, 128, 0],
            type='',
            swap=''),
   
    },
    skeleton_info={
        0:
        dict(link=('LUF', 'RUF'), id=0, color=[0, 255, 0]),
        # 1:
        # dict(link=('LUF', ''), id=1, color=[0, 255, 0]),
        # 2:
        # dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        # 3:
        # dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        # 4:
        # dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        # 5:
        # dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        # 6:
        # dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        # 7:
        # dict(
        #     link=('left_shoulder', 'right_shoulder'),
        #     id=7,
        #     color=[51, 153, 255]),
        # 8:
        # dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        # 9:
        # dict(
        #     link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        # 10:
        # dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        # 11:
        # dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        # 12:
        # dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        # 13:
        # dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        # 14:
        # dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        # 15:
        # dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        # 16:
        # dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        # 17:
        # dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        # 18:
        # dict(
        #     link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 
        # 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        # 1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 
        # 0.062,
        # 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])
