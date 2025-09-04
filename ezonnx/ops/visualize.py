from typing import List,Optional,Union
import numpy as np
import matplotlib.pyplot as plt
def visualize_images(titles:Union[List[str],str], 
                     images:Union[List[np.ndarray],np.ndarray]):
    """Visualize multiple images in a single row.

    Args:
        titles (Union[List[str],str]): List of titles or title for each image.
        images (Union[List[np.ndarray],np.ndarray]): List of BGR images or image to display.
    """
    if isinstance(titles, str):
        titles = [titles]
    if isinstance(images, np.ndarray):
        images = [images]
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1:
        axes = [axes]
    
    # Ensure we have the same number of titles as images
    if len(titles) < len(images):
        titles = titles + [''] * (len(images) - len(titles))
    for ax, title, image in zip(axes, titles, images):
        if len(image.shape) == 2:
            ax.imshow(image, cmap='plasma')
        else:
            ax.imshow(image[..., ::-1])  # Convert BGR to RGB for displaying
        ax.set_title(title)
        ax.axis('off')
    plt.show()

from IPython.display import clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
def show_3d_poses(poses_3d,clear=True):
    # H36M形式の関節の接続情報（インデックスで定義）
    connections = [
        (0, 1), (1, 2), (2, 3),  # 右脚
        (0, 4), (4, 5), (5, 6),  # 左脚
        (0, 7), (7, 8), (8, 9),(9, 10),  # 胴体と頭
        (8, 11), (11, 12), (12, 13),  # 左腕
        (8, 14), (14, 15), (15, 16)   # 右腕
    ]

    # 3Dプロットの作成
    deg=180
    for pose in np.array(poses_3d):
        if clear:
            clear_output(wait=True)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 3Dプロット用にprojection='3d'を追加
        
        # 関節点をプロット
        ax.scatter( pose[:,2],pose[:,0], pose[:,1], color='blue')  # ｚ, x, yの順にプロット
        
        # 各関節を線で繋ぐ
        for k,(i, j) in enumerate(connections):
            if k<3:
                color="blue"# 右脚
            if k>=3 and k<6:
                color="red"  # 左脚
            if k>=6 and k<10:
                color="black" # 胴体と頭
            if k>=10 and k<13:
                color="pink"  # 左腕
            if k>=13:
                color="green"  # 右腕
            # 可視化で見やすくするため軸に対するデータのxyzを入れ替えている
            ax.plot([pose[i, 2], pose[j, 2]],  # Z軸の座標
                [pose[i, 0], pose[j, 0]],  # X軸の座標
                    [pose[i, 1], pose[j, 1]],  # Y軸の座標
                    color=color)
        
        # 軸の設定
        ax.set_xlim([-1, 1])
        ax.set_ylim([1, -1])
        ax.set_zlim([1, -1])
        # 可視化で見やすくするため軸に対するデータのxyzを入れ替えている
        ax.set_xlabel('Z axis')
        ax.set_ylabel('X axis')
        ax.set_zlabel('Y axis')
        

        ax.view_init(elev=20, azim=deg)
        # ax.axis("off")
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        # time.sleep(0.05)
        deg+=1%360
        
        plt.show()