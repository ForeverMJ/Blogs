# Player Contact Detection Challenge—Getting Started

## **NFL - Player Contact Detection Challenge**

![image.png](Player%20Contact%20Detection%20Challenge%E2%80%94Getting%20Started%20c8f3186b27e84377aeda24209395d922/image.png)

在这个竞赛中，利用比赛镜头和追踪数据，我们要预测一对运动员之间的接触时刻，还有运动员什么时候对地面进行非脚接触（non-foot contact）。每一轮有四个相关的视频。两个视频展示了边线和底线视角，是时间同步和对齐的。另外，一个 AII29 view不能保证时间同步。训练视频在 train/train_labels.csv，用来预测的测试视频在 **test/** 文件夹下。

**train_baseline_helmets.csv：是去年赢得胜利运动员的分配模型。**

**train_player_tracking.csv：提供了10HZ的追踪数据**

**train_video_metadata.csv：包括边线和底线视角的时间戳同步的追踪数据。**

```python
import numpy as np  //用于数值计算，尤其是数组和矩阵操作
import pandas as pd //用于数据处理和分析，数据label操作
import matplotlib.pylab as plt //数据可视化

from sklearn.metrics import matthews_corrcoef //matthews_corrcoef用于评估分类模型的性能

# Read in data files
BASE_DIR = "../input/nfl-player-contact-detection" //数据文件所在路径

# Labels and sample submission 从train_labels.csv中读取训练标签数据，并将其中的datatime列解析为日期时间格式
labels = pd.read_csv(f"{BASE_DIR}/train_labels.csv", parse_dates=["datetime"])

ss = pd.read_csv(f"{BASE_DIR}/sample_submission.csv") //读取实例提交文件

# Player tracking data
tr_tracking = pd.read_csv(
    f"{BASE_DIR}/train_player_tracking.csv", parse_dates=["datetime"] //读取训练集中的球员跟踪数据
)
te_tracking = pd.read_csv(
    f"{BASE_DIR}/test_player_tracking.csv", parse_dates=["datetime"] //读取测试集中的球员跟踪数据
)

# Baseline helmet detection labels
tr_helmets = pd.read_csv(f"{BASE_DIR}/train_baseline_helmets.csv") //读取头盔检测的训练集
te_helmets = pd.read_csv(f"{BASE_DIR}/test_baseline_helmets.csv") //读取头盔检测的测试集

# Video metadata with start/stop timestamps //读取视频元数据
tr_video_metadata = pd.read_csv(
    "../input/nfl-player-contact-detection/train_video_metadata.csv",
    parse_dates=["start_time", "end_time", "snap_time"],  //将start_time", "end_time", "snap_time列解析为日期时间格式
)
```

**What is the goal of this competition?**

简单来说，我们要尝试预测每0.1秒中，有没有接触发生。

Contact可以有两种形式：1️⃣ 两个运动员之间的接触 2️⃣ 运动员和地面接触（Note：当运动员除了手或脚之外的任何情况接触地面）

提交应该包括22个运动员所有可能性的预测，以及每个运动员和地面的接触。每个预测对只需要一行，较低玩家的nfl_player_id为nfl_player_id_1，较大玩家为nfl_player_id_2。

### **Video and Baseline Boxes**

下面是一个例子，展示了基线预测的实例以及真实的头盔标签。

```python
import os  //用于与操作系统交互，进行文件路径操作和文件删除
import cv2 //OPENCV库，用于图像处理和视频处理
import subprocess //在python中执行系统命令，用于调用‘ffmpeg’工具进行视频编码转换
from IPython.display import Video, display //用于在Jupyter Notebook中显示视频
import pandas as pd //用于处理DataFrame

//三个参数：1视频路径，2检测框，3是否打印处理过程的信息
def video_with_helmets(
    video_path: str, baseline_boxes: pd.DataFrame, verbose=True
) -> str:     
    """
    Annotates a video with baseline model boxes and labels.
    """
    //主要逻辑
    //常量和初始化
    VIDEO_CODEC = "MP4V" //指定编码格式
    HELMET_COLOR = (0, 0, 0)  # Black //定义头盔颜色是黑色
    video_name = os.path.basename(video_path) //提取视频文件的名称
    if verbose:
        print(f"Running for {video_name}") //打印正在处理的视频名称
    baseline_boxes = baseline_boxes.copy() //创建副本，以免改变原始数据

    vidcap = cv2.VideoCapture(video_path) //打开视频文件进行处理
    fps = vidcap.get(cv2.CAP_PROP_FPS)  //获取视频帧率
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) //获取视频宽度
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) //获取视频高度
    output_path = "labeled_" + video_name //生成的标注视频的输出路径
    tmp_output_path = "tmp_" + output_path //临时输出路径，稍后会用‘ffmpeg’对视频进行进一步处理
    output_video = cv2.VideoWriter(
        tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height)
    ) //初始化视频写入器，用于将处理后的帧写入新的视频文件。

		//处理每一帧
    frame = 0  //计数帧数，初始为0
    while True: 循环中，逐帧读取视频，img是图像，it_worked表示读帧是否成功
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        # We need to add 1 to the frame count to match the label frame index
        # that starts at 1
        frame += 1

        # Let's add a frame index to the video so we can track where we are
        //在视频帧上绘制框和标签
        img_name = video_name.replace(".mp4", "")
        //在帧的左上角绘制视频名称
        cv2.putText(
            img,
            img_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            HELMET_COLOR,
            thickness=1,
        )
        //在帧的右下角绘制当前帧的编号
        cv2.putText(
            img,
            str(frame),
            (1280 - 90, 720 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            HELMET_COLOR,
            thickness=1,
        )

        # Now, add the boxes
        //query方法从baseline_boxes中筛选出当前帧的检测框数据
        boxes = baseline_boxes.query("video == @video_name and frame == @frame")
        for box in boxes.itertuples(index=False):
        //对于每个检测框，绘制矩形框和标签
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                HELMET_COLOR,
                thickness=1,
            )
            cv2.putText(
                img,
                box.player_label,
                (box.left + 1, max(0, box.top - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                HELMET_COLOR,
                thickness=1,
            )
				//将处理好的帧写入输出视频文件
				output_video.write(img)
				//释放写入器
        output_video.release()
    # Not all browsers support the codec, we will re-load the file at tmp_output_path
    
    # and convert to a codec that is more broadly readable using ffmpeg
    //使用ffmpeg转换视频编码为H.264编码格式，并输出到output_path
    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            tmp_output_path,
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-hide_banner",
            "-loglevel",
            "error",
            "-vcodec",
            "libx264",
            output_path,
        ]
    )
    os.remove(tmp_output_path) //删除临时视频文件

    return output_path //函数返回生成的标注的视频文件路径
```

下面视频显示了在一个回合中这些检测框。并在检测框的上方显示主客队以及球员号码。

```python
//使用之前定义的video_with_helmets函数来处理一个接触检测的实例视频，并在Jupyter Notebook中显示生成的带有头盔检测框的标注视频
example_video = "../input/nfl-player-contact-detection/train/58168_003392_Sideline.mp4"
//tr_helmets：包含训练数据中的头盔检测框数据的 pandas.DataFrame
output_video = video_with_helmets(example_video, tr_helmets)

frac = 0.65  # scaling factor for display 视频缩放比例
//Jupyter中显示视频。data=output_video视频文件的路径，embed=true：将视频嵌入到NoteBook显示。。。
display(
    Video(data=output_video, embed=True, height=int(720 * frac), width=int(1280 * frac))
)
```

![image.png](Player%20Contact%20Detection%20Challenge%E2%80%94Getting%20Started%20c8f3186b27e84377aeda24209395d922/image%201.png)

### **NGS Tracking Data**

使用NGS跟踪数据对于正确标记视频非常重要。需要注意的一些事情是：

1. NGS数据以10Hz的速度采样，而视频以大约59.94Hz的速度采样。NGS数据是每秒10次进行采样的，而视频是每秒接近60帧画面，这意味着视频帧和NGS数据不是一一对应的，因此需要某种方式进行同步。
2. 跟踪数据可以使用训练元数据与视频大致同步。（训练元数据包含视频的开始，结束时间，快照时间等信息，这些可以帮助NGS数据中的每个步骤对应到正确的视频帧中）
3. NGS数据包含一个step列，这个列表示每个时间步的编号。我们可以用它来与标签和样本提交文件相结合。
4. NGS数据还包括速度、加速度、方向等特征，可以为训练模型提供更多的上下文信息，提升模型的性能和预测精度。

定义一个函数create_football_field，用于绘制一个美式橄榄球场地的图像，适用于对球赛进行可视化处理

```python
import matplotlib.patches as patches
import matplotlib.pylab as plt

def create_football_field(
    linenumbers=True, //是否显示场地上的码数编号
    endzones=True,    //是否显示两个端区
    figsize=(12, 6.33), //定义图像尺寸
    line_color="black", //场地线条的颜色
    field_color="white", //场地背景颜色
    ez_color=None,       //端区的颜色，如果不指定则使用场地颜色
    ax=None,             //传入的绘图区域，如果为None，则创建新的绘图区域
    return_fig=False,     //控制函数返回内容，True时返回fig和ax，否则仅返回ax
):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """

    if ez_color is None:
        ez_color = field_color

		//创建场地矩形区域，使用patches.Rectangle创建一个矩形
    rect = patches.Rectangle(
        (0, 0),
        120,
        53.3,
        linewidth=0.1,
        edgecolor="r",  //边框颜色红色
        facecolor=field_color, //填充颜色为白色
        zorder=0,
    )
		//如果没有传入绘图区域ax，则使用plt.subplots创建一个新的图像，并将创建的矩形添加到这个图像中
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)
    //使用ax.plot方法绘制橄榄球场的线条
    ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color=line_color)
    
    //绘制端区，端区长度为10，宽度与场地一致
    # Endzones
    if endzones:
        ez1 = patches.Rectangle(
            (0, 0),
            10,
            53.3,
            linewidth=0.1,
            edgecolor=line_color,
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ez2 = patches.Rectangle(
            (110, 0),
            10,
            53.3,
            linewidth=0.1,
            edgecolor=line_color,
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ax.add_patch(ez1)
        ax.add_patch(ez2)

    ax.axis("off")
    //绘制场地数字
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            ax.text(
                x,
                5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color=line_color,
            )
            ax.text(
                x - 0.95,
                53.3 - 5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color=line_color,
                rotation=180,
            )
            
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color=line_color)
        ax.plot([x, x], [53.0, 52.5], color=line_color)
        ax.plot([x, x], [22.91, 23.57], color=line_color)
        ax.plot([x, x], [29.73, 30.39], color=line_color)
    //绘制场地边界
    border = patches.Rectangle(
        (-5, -5),
        120 + 10,
        53.3 + 10,
        linewidth=0.1,
        edgecolor="orange",
        facecolor=line_color,
        alpha=0,
        zorder=0,
    )
    ax.add_patch(border)
    //设定图像显示范围，设定场地的x轴和y轴的范围，确保整个橄榄球场都在图像中显示。
    ax.set_xlim((-5, 125))
    ax.set_ylim((-5, 53.3 + 5))

		//如果return_fig为True，则返回图像和绘图区域对象，否则只返回绘图区域对象
    if return_fig:
        return fig, ax
    else:
        return ax
```

```python
//可视化特定时刻球员位置的功能
game_play = "58168_003392"  //选择特定比赛的标识符
//从tr_tracking数据中，使用query方法选择特定的比赛并处于步数为0（通常是球赛的起点）的球员的追踪数据
example_tracks = tr_tracking.query("game_play == @game_play and step == 0")
ax = create_football_field() //ax是绘图的轴对象
for team, d in example_tracks.groupby("team"):  //为每只球队绘制球员位置
		//对于每个球队的球员数据，使用ax.scatter函数绘制球员的位置信息。
    ax.scatter(
        d["x_position"], //球员在场地上的横坐标
        d["y_position"], //球员在场地上的纵坐标
        label=team,      //每个球员点的标签为所属球队
        s=65,            //设置散点大小
        lw=1,            //设置球员点的边框宽度
        edgecolors="black", //设置球员点的边框三色
        zorder=5,         //确保散点绘制在较高的层级上，不被其他元素覆盖
    )
ax.legend().remove()  //移除图例
ax.set_title(f"Tracking data for {game_play}: at step 0", fontsize=15) //为图像添加一个标题，显示当前游戏标识符和步数信息。fontsize=15 用于设置标题字体的大小
plt.show() //显示图像
```

![image.png](Player%20Contact%20Detection%20Challenge%E2%80%94Getting%20Started%20c8f3186b27e84377aeda24209395d922/image%202.png)

```python
tr_tracking.head() //显示追踪数据的前5行
```

![image.png](Player%20Contact%20Detection%20Challenge%E2%80%94Getting%20Started%20c8f3186b27e84377aeda24209395d922/image%203.png)

### **Video Metadata**

这些文件提供了可用于将视频文件与NGS跟踪数据同步的信息。

![image.png](Player%20Contact%20Detection%20Challenge%E2%80%94Getting%20Started%20c8f3186b27e84377aeda24209395d922/image%204.png)

### **Contact Labels**

Contact Labels这个标签识别了两名球员接触或者球员接触地面的时刻

标签以10Hz的频率提供，并且与NGS追踪数据同步。NGS数据指的是球员追踪数据，采样率是10Hz。

时间范围：标签的开始时间是step=0，结束时间和最短的视频（侧面或端区视频）相关的step

每一行代表一种接触场景：

**球员之间的接触**：每一对球员的组合都会被记录，其中 `nfl_player_id_1` 总是小于 `nfl_player_id_2`。这意味着数据中不会出现反向的重复组合（例如，`nfl_player_id_1 < nfl_player_id_2`）。

**球员与地面接触**：如果球员接触地面，则 `nfl_player_id` 会被标记为 `"G"`。

`contact_id` 列是每个接触事件的唯一标识符，它是 `game_play`、`step`、`nfl_player_id_1` 和 `nfl_player_id_2` 的组合。

`contact` 列是一个二进制值，其中 `0` 表示没有接触，`1` 表示发生接触。

尽管标签和视频以不同的速率采样，但我们大致可以将它们连接起来，以便我们可以将它们与视频一起可视化。下面我们提供了一个加入的示例功能。

![image.png](Player%20Contact%20Detection%20Challenge%E2%80%94Getting%20Started%20c8f3186b27e84377aeda24209395d922/image%205.png)

这段代码定义了一个 `join_helmets_contact` 函数，用于将头盔检测数据和接触标签数据结合在一起，以便于可视化接触事件。

函数参数：

- `game_play`: 特定的一段比赛
- `labels`: 接触标签的数据集，标记了球员之间或球员与地面之间的接触事件。
- `helmets`: 头盔检测数据集，表示视频中的头盔框。
- `meta`: 元数据集，包括视频的时间戳等信息。
- `view`: 视频视角，默认为 "Sideline"。
- `fps`: 视频的帧率，默认值为 59.94 帧每秒。

```python
def join_helmets_contact(game_play, labels, helmets, meta, view="Sideline", fps=59.94):
    """
    Joins helmets and labels for a given game_play. Results can be used for visualizing labels.
    Returns a dataframe with the joint dataframe, duplicating rows if multiple contacts occur.
    """
    //根据 game_play 从标签和头盔数据集中筛选出相关的行，并进行复制，避免修改原数据。
    gp_labs = labels.query("game_play == @game_play").copy()
    gp_helms = helmets.query("game_play == @game_play").copy()
    
		//根据 game_play 和 view 从元数据中提取视频的开始时间。
    start_time = meta.query("game_play == @game_play and view == @view")[
        "start_time"
    ].values[0]

		//将头盔数据中的 frame 信息转换为实际时间戳（datetime），方法是根据帧率将帧数转换为时间间隔，并加上视频的开始时间。
    gp_helms["datetime"] = (
        pd.to_timedelta(gp_helms["frame"] * (1 / fps), unit="s") + start_time
    )
    gp_helms["datetime"] = pd.to_datetime(gp_helms["datetime"], utc=True)
    
    //为了使视频帧与NGS数据同步，将 datetime 列调整为 datetime_ngs，并向前推进 50 毫秒，然后将其四舍五入为 100 毫秒精度。这一步是为了近似对齐不同采样率的数据
    gp_helms["datetime_ngs"] = (
        pd.DatetimeIndex(gp_helms["datetime"] + pd.to_timedelta(50, "ms"))
        .floor("100ms")
        .values
    )
    gp_helms["datetime_ngs"] = pd.to_datetime(gp_helms["datetime_ngs"], utc=True)
    
		//将接触标签的 datetime 转换为UTC格式，并创建与NGS追踪数据同步的时间列 datetime_ngs。
    gp_labs["datetime_ngs"] = pd.to_datetime(gp_labs["datetime"], utc=True)
    
		//将头盔数据和接触标签数据合并在一起，基于 datetime_ngs 和球员ID进行连接（即 nfl_player_id 和 nfl_player_id_1 匹配）。只保留接触事件（contact == 1）
    gp = gp_helms.merge(
        gp_labs.query("contact == 1")[
            ["datetime_ngs", "nfl_player_id_1", "nfl_player_id_2", "contact_id"]
        ],
        left_on=["datetime_ngs", "nfl_player_id"],
        right_on=["datetime_ngs", "nfl_player_id_1"],
        how="left",
    )
    //返回合并后的数据框gp
    return gp
```

这段代码定义了一个 `video_with_contact` 函数，用于在视频中绘制球员的头盔检测框，并通过不同的颜色标记球员之间的接触状态。该函数主要用于生成一个带有标注的输出视频

```python
import os  //导入 os 模块，用于与操作系统交互，例如文件路径和文件操作。
import cv2 //导入 cv2 模块，这是 OpenCV 库，用于图像和视频处理。
import subprocess  //导入 subprocess 模块，用于运行外部程序（例如 ffmpeg）来处理视频转换。
from IPython.display import Video, display //从 IPython 中导入 Video 和 display，用于在 Jupyter Notebook 中显示视频。 
import pandas as pd //导入 pandas 库，并将其命名为 pd，用于数据处理和分析。

//函数参数：
//video_path: 输入视频的文件路径。
//baseline_boxes: 一个包含球员头盔位置和接触信息的数据框（pandas.DataFrame）。
//verbose: 是否打印日志信息，默认为 True。
def video_with_contact(
    video_path: str, baseline_boxes: pd.DataFrame, verbose=True
) -> str:
    """
    Annotates a video with baseline model boxes.
    Helmet boxes are colored based on the contact label.
    """
    VIDEO_CODEC = "MP4V"  //定义视频编码
    HELMET_COLOR = (0, 0, 0)  # Black //头盔框的默认颜色 黑色
    video_name = os.path.basename(video_path) //从 video_path 中提取视频名称
    if verbose:
        print(f"Running for {video_name}")
    baseline_boxes = baseline_boxes.copy() //生成副本，不更改原文件

    vidcap = cv2.VideoCapture(video_path) //使用 OpenCV 的 cv2.VideoCapture 打开输入视频，并获取视频的帧率、宽度和高度。
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = "contact_" + video_name //准备输出视频的路径
    tmp_output_path = "tmp_" + output_path //准备输出视频的***临时路径**
    //使用 OpenCV 的 cv2.VideoWriter 初始化一个视频写入器，用于写入处理后的视频
    //tmp_output_path: 临时文件路径。
		//cv2.VideoWriter_fourcc(*VIDEO_CODEC): 视频编码格式。
		//fps: 视频帧率。
		//(width, height): 视频帧的尺寸。*
    output_video = cv2.VideoWriter(
        tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height)
    )
    frame = 0 //初始化帧数计数器为0
    //无限循环，逐帧读入视频
    while True:
        it_worked, img = vidcap.read() //从视频中读取一帧，img 是帧的图像数据，it_worked 是读取成功与否的标志。
        if not it_worked: //如果帧读取失败（视频结束），退出循环
            break
        # We need to add 1 to the frame count to match the label frame index
        # that starts at 1
        frame += 1  //每读取一帧，frame 计数器加 1

        # Let's add a frame index to the video so we can track where we are
        img_name = video_name.replace('.mp4','') //从视频文件名中去掉 .mp4 扩展名。
        cv2.putText( //在图像 img 上绘制文本
            img,
            img_name, //要绘制的文本内容
            (10, 30), //文本在图像上的位置（左上角的坐标
            cv2.FONT_HERSHEY_SIMPLEX, //字体样式
            1,
            HELMET_COLOR, //文本颜色（黑色）
            thickness=1,  //字体粗细
        )
        
        cv2.putText( //在图像的右下角绘制当前帧编号
            img,
            str(frame),
            (1280 - 90, 720 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            HELMET_COLOR,
            thickness=1,
        )

        # Now, add the boxes
        // 从 baseline_boxes 中筛选出与当前视频和当前帧匹配的头盔检测框。
        boxes = baseline_boxes.query("video == @video_name and frame == @frame")
        // 筛选出涉及接触的球员（nfl_player_id_2 非空且不为 "G"，表示球员与球员的接触）
        contact_players = boxes.dropna(subset=["nfl_player_id_2"]).query(
            'nfl_player_id_2 != "G"'
        )
        //将接触的球员 ID 转换为整数列表，以便后续用于判断球员之间的接触。
        contact_ids = (
            contact_players["nfl_player_id_1"].astype("int").values.tolist()
            + contact_players["nfl_player_id_2"].astype("int").values.tolist()
        )
        for box in boxes.itertuples(index=False): //使用 itertuples() 遍历每一个头盔检测框的行，index=False 表示不返回行的索引。
						//判断头盔检测框的接触情况
            if box.nfl_player_id_2 == "G": //如果球员与地面接触（nfl_player_id_2 == "G"），将框的颜色设为红色，线条粗细设为 2。
                box_color = (0, 0, 255)  # Red
                box_thickness = 2
            elif int(box.nfl_player_id) in contact_ids: //如果球员与其他球员接触，将框的颜色设为绿色，线条粗细设为 2。
                box_color = (0, 255, 0)  # green
                box_thickness = 2
								//如果当前球员与另一球员接触，在两者之间绘制一条蓝色线条，表示接触关系
                # Add line between players in contact
                /*这里 box.nfl_player_id_2 表示当前 box（头盔框）中与 nfl_player_id_1 接触的另一名球员的 ID。
np.isnan(float(box.nfl_player_id_2)) 检查 nfl_player_id_2 是否为 NaN（即不存在的值）。因为 NaN 意味着当前球员没有与其他球员接触，
所以如果 box.nfl_player_id_2 不是 NaN，则说明这名球员与另一名球员发生了接触，继续执行代码。*/
                if not np.isnan(float(box.nfl_player_id_2)):
                    player2 = int(box.nfl_player_id_2) //player2 取自 box.nfl_player_id_2，代表与当前球员接触的另一名球员的 ID，并将其转换为整数
                    player2_row = boxes.query("nfl_player_id == @player2") //player2_row 是一个 DataFrame，通过 query 函数从 boxes（当前帧的所有检测框）中查找 nfl_player_id 等于 player2 的那一行。这一行对应的就是球员2在当前帧中的头盔位置数据
                    if len(player2_row) == 0: //如果 player2_row 为空（即长度为 0），表示 player2（与当前球员接触的另一名球员）在当前帧中并没有被检测到，可能是因为该球员不在视野中或没有检测到该球员
                        # Player 2 is not in view
                        continue //这时候代码跳过这个 if 分支，并进入下一个循环（即不绘制线条），因为无法绘制两名球员之间的线条
                    cv2.line( //cv2.line 是 OpenCV 中的一个函数，用于在图像 img 上绘制一条线
                        img, 
                        (box.left + int(box.width / 2), box.top + int(box.height / 2)),   //起点
                        (
                            player2_row.left.values[0]  //终点
                            + int(player2_row.width.values[0] / 2),
                            player2_row.top.values[0]
                            + int(player2_row.height.values[0] / 2),
                        ),
                        color=(255, 0, 0),
                        thickness=2,
                    )
						//如果没有接触，框的颜色为默认的黑色，线条粗细设为 1
            else:
                box_color = HELMET_COLOR
                box_thickness = 1

            # Draw lines between two boxes
						//在头盔检测位置绘制矩形框，框的颜色和粗细依据接触情况而定。
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                box_color,
                thickness=box_thickness,
            )
            cv2.putText( //在框的上方绘制球员的标签（如球员编号）
                img,
                box.player_label,
                (box.left + 1, max(0, box.top - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                HELMET_COLOR,
                thickness=1,
            )

        output_video.write(img) //将处理后的图像帧写入输出视频文件
    output_video.release() //释放视频写入器资源，关闭输出视频文件
    # Not all browsers support the codec, we will re-load the file at tmp_output_path
    # and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(output_path):
        os.remove(output_path)
        //使用 ffmpeg 将临时视频文件转换为兼容性更好的 libx264 编码格式，并保存为最终的输出文件。
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            tmp_output_path,
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-hide_banner",
            "-loglevel",
            "error",
            "-vcodec",
            "libx264",
            output_path,
        ]
    )
    //删除临时视频文件
    os.remove(tmp_output_path)
    //返回最终输出视频的路径
    return output_path
```

### **Example Video with Labels**

下面一段视频显示了头盔检测之间的关联：

黑色检测框代表运动员没有被接触，而且会显示一个唯一的号码。

绿色检测框代表运动员被一名或多名运动员接触

红色检测框代表运动员接触地面（也有可能是其他运动员）

蓝线代表互相接触的两个运动员。

```python
game_play = "58168_003392" //设置比赛ID
//调用 join_helmets_contact 函数，结合头盔的识别数据和接触标签，为特定的比赛 (game_play) 生成包含接触信息的数据。
gp = join_helmets_contact(game_play, labels, tr_helmets, tr_video_metadata)

example_video = f"../input/nfl-player-contact-detection/train/{game_play}_Sideline.mp4"
//这个函数用于在视频中为每一帧添加接触标记框。传入的参数是视频文件路径 (example_video) 和包含接触信息的数据 (gp)。
output_video = video_with_contact(example_video, gp)

frac = 0.65  # scaling factor for display //定义一个缩放比例，用于调整视频展示的尺寸
display( //显示
    Video(data=output_video, embed=True, height=int(720 * frac), width=int(1280 * frac))
)
```

![image.png](Player%20Contact%20Detection%20Challenge%E2%80%94Getting%20Started%20c8f3186b27e84377aeda24209395d922/image%206.png)

### **Baseline Approach**

其实我们可以创建一个简单的方式——-仅依赖于追踪数据，旨在根据训练数据为比赛指标识别出一个最佳的阈值。

该基线模型的核心思想是通过计算球员之间的距离来预测是否存在接触，并使用某种距离阈值来判断接触与否。这里不考虑球员与地面的接触，默认将所有球员与地面的接触预测为“无接触”

### 具体过程如下：

1. **计算球员之间的分离距离**：
    
    对于每个 `contact_id`（唯一标识符，包含比赛标识符、时间步、两名球员的 ID），计算两个球员之间的分离距离。如果两个球员之间的距离在某个阈值之下，则判定为接触。
    
2. **处理球员与地面的接触**：
    
    对于涉及球员与地面接触的行（`nfl_player_id_2 == "G"`），将其分离距离填充为 99 码（即一个很大的距离），这样它们在模型中被处理为非接触情况。
    
3. **遍历阈值并计算指标**：
    
    在一个固定的范围（例如 0 到 5 码之间），选择不同的阈值，计算模型在训练集上的表现，使用比赛的评估指标来衡量预测效果。
    
4. **选择最佳阈值**：
    
    找到能够产生最佳分数的阈值，并将这个阈值应用到测试集的提交中。这个阈值即是最能区分接触和非接触的距离界限。
    

这个方法的简单之处在于它没有复杂的特征工程或模型设计，仅依赖于球员之间的物理距离，进而通过阈值的选择来优化结果。这种基线方法虽然简单，但提供了一个可行的出发点，以后可以在此基础上进行优化和改进。

```python
//计算两名球员在比赛中的分离距离。
def compute_distance(df, tr_tracking, merge_col="datetime"):
    """
    Merges tracking data on player1 and 2 and computes the distance.
    """
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id", "x_position", "y_position"]
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={"x_position": "x_position_1", "y_position": "y_position_1"})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id", "x_position", "y_position"]
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={"x_position": "x_position_2", "y_position": "y_position_2"})
        .copy()
    )

    df_combo["distance"] = np.sqrt(
        np.square(df_combo["x_position_1"] - df_combo["x_position_2"])
        + np.square(df_combo["y_position_1"] - df_combo["y_position_2"])
    )
    return df_combo

//该函数的作用是为每个接触记录创建一个唯一的 contact_id。
def add_contact_id(df):
    # Create contact ids
    df["contact_id"] = (
        df["game_play"]
        + "_"
        + df["step"].astype("str")
        + "_"
        + df["nfl_player_id_1"].astype("str")
        + "_"
        + df["nfl_player_id_2"].astype("str")
    )
    return df

//该函数用于将 contact_id 拆分为独立的列，方便后续分析。
def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df
//通过计算球员之间的距离并合并步数信息，将这些距离数据转换为一个可供进一步分析的结构化 DataFrame   
df_combo = compute_distance(labels, tr_tracking)

print(df_combo.shape, labels.shape) //查看数据合并后的大小是否符合预期，从而确认距离计算是否正确完成。

df_dist = df_combo.merge(
    tr_tracking[["game_play", "datetime", "step"]].drop_duplicates()
)
df_dist["distance"] = df_dist["distance"].fillna(99)  # Fill player to ground with 99
```

```python
//通过计算球员之间的距离和二分类结果的马修斯相关系数（MCC）来评估预测模型的性能
for dist in range(0, 5): //目标是通过不同的距离阈值来评估模型的性能，找出最佳阈值。
    score = matthews_corrcoef(df_dist["contact"], df_dist["distance"] <= dist)
    print(f"Threshold Distance: {dist} Yard - MCC Score: {score:0.4f}")
    
    
Threshold Distance: 0 Yard - MCC Score: 0.0000
Threshold Distance: 1 Yard - MCC Score: 0.5201 ---）最佳阈值
Threshold Distance: 2 Yard - MCC Score: 0.3592
Threshold Distance: 3 Yard - MCC Score: 0.2517
Threshold Distance: 4 Yard - MCC Score: 0.1935
```

- `matthews_corrcoef` 是用于二分类问题的一个评价指标。它根据混淆矩阵的值（TP、TN、FP、FN）计算马修斯相关系数（MCC），该指标的取值范围是 [-1, 1]，值越接近 1 表示预测效果越好。
- **`df_dist["contact"]`**: 这是模型的真实标签，包含了接触（1）或无接触（0）的二分类信息。
- **`df_dist["distance"] <= dist`**: 这是模型的预测结果。这里我们根据球员之间的距离来进行预测，如果两名球员之间的距离小于等于 `dist`（当前阈值），则预测为 1（表示接触）；否则预测为 0（表示无接触）。
- 通过将 `df_dist["contact"]` 与 `df_dist["distance"] <= dist` 进行对比，`matthews_corrcoef` 会计算出模型在当前距离阈值下的 MCC 分数。

### **Create Baseline Submission**

上面我们已经计算出最佳阈值是1，下面我们来生成比赛提交的文件。

```python
ss = pd.read_csv(f"{BASE_DIR}/sample_submission.csv") //读取包含比赛的样例提交格式的CSV文件

THRES = 1 //阈值为1yard

ss = expand_contact_id(ss) 
ss_dist = compute_distance(ss, te_tracking, merge_col="step") //结合追踪数据（te_tracking），计算每个 contact_id 对应的球员之间的距离。

print(ss_dist.shape, ss.shape)

submission = ss_dist[["contact_id", "distance"]].copy() //创建一个新的数据框 submission，仅包含 contact_id 和计算出的 distance 列
submission["contact"] = (submission["distance"] <= THRES).astype("int") //根据距离阈值 THRES，如果两名球员之间的距离小于等于 1 码，则标记 contact 为 1（表示有接触）；否则标记为 0（表示无接触）。
submission = submission.drop('distance', axis=1)  //删除 distance 列，只保留 contact_id 和 contact 列，以符合比赛的提交格式
submission[["contact_id", "contact"]].to_csv("submission.csv", index=False) //将最终的 submission 数据框保存为 submission.csv 文件。index=False 表示不保存索引列。

submission.head()
```

**End**；