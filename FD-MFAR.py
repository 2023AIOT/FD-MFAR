import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math
from scipy.signal import savgol_filter  # type: ignore
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 设置matplotlib字体，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 过滤字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tkinter')

# 通信参数
ACTIVE_PERIOD = 0.5  # 活动期占比
CONTENTION_PERIOD = 0.3  # 争用访问期占活动期的比例
SLOT_TIME = 0.001  # 时隙长度(s)
MAX_BACKOFF = 3 # 最大退避次数
CW_MIN = 4  # 最小竞争窗口
CW_MAX = 256  # 最大竞争窗口
DIFS = 0.001  # 设备开始发送数据之前需要等待的空闲时间，用于确保信道空闲。
SIFS = 0.0005  # 用于设备之间的控制信息交换的时间间隔
FRAME_TIME = 0.1    # TDMA帧时间 100ms

# 能量和休眠参数
DUTY_CYCLE_MIN = 0.2  # 设备在一个周期内至少有20%的时间处于活动状态
DUTY_CYCLE_MAX = 0.8  # 设备在一个周期内至多有80%的时间处于活动状态
LOAD_THRESHOLD_LOW = 0.3  # 设备负载低于30%时，进入低功耗模式或其他节能模式
LOAD_THRESHOLD_HIGH = 0.7  # 设备负载高于70%时，进入高功耗模式
SLEEP_POWER = 0.001  # 睡眠状态能耗比例

# HEED特有参数
HEED_Cprob = 0.07  # 初始簇头概率
HEED_Pmin = 0.001  # 最小簇头概率
HEED_ITERATIONS = 2  # HEED迭代次数
HEED_COST_FACTOR = 0.6  # 代价函数中能量权重

# D2CRP特有参数
D2CRP_BROADCAST_RANGE = 15
D2CRP_P = 0.06

# 添加新的能耗相关常量
BUFFER_POWER = 0.0000001  # 数据缓存能耗系数
WAKEUP_POWER = 0.0001  # 唤醒切换能耗
SLEEP_TRANSITION_POWER = 0.00005  # 休眠切换能耗

# 网络参数常量
MAX_BUFFER_SIZE = 1000  # 节点最大缓冲区大小(packets)
MAX_SNR = 100  # 最大信噪比(dB)

#定义全局参数
HARD_THRESHOLD = 0.7
SOFT_THRESHOLD = 0.3
MAX_WAIT = 50  # 最大等待轮数
ALPHA = 0.5  # 数据变化权重
BETA = 0.5  # 偏离均值权重

# === 轻唤醒机制参数 ===
CONFIRM_WINDOW = 0.02  # 轻唤醒确认窗口（单位：秒，仿真可用轮数代替）
CONFIRM_THRESHOLD = 2 # 至少2个邻居确认

# 添加新的常量定义
DELAY_WINDOW_SIZE = 10  # 时延检测滑动窗口大小
DELAY_THRESHOLD_FACTOR = 2.0  # 时延异常阈值系数
MAX_REROUTE_HOPS = 2  # 最大重路由跳数范围
PARALLEL_PATH_THRESHOLD = 0.8  # 触发并行传输的负载阈值

# === MAC调度对比实验参数 ===
MAC_MODE = {
    'TDMA': 'TDMA',
    'CSMA_CA': 'CSMA_CA',
    'HYBRID': 'HYBRID',
    'FDMMBF-ARP': 'FDMMBF-ARP'
}

# 数据优先级阈值（用于混合模式）
PRIORITY_THRESHOLD = 0.7  # 高于此值使用CSMA/CA

# === 图表配色方案 ===
PLOT_COLORS = {
    'color1': '#EFFCEE',  # 浅绿色
    'color2': '#D4EEFF',  # 浅蓝色
    'color3': '#FFE5D6',  # 浅橙色
    'color4': '#FFF2CD',  # 浅黄色
    'color5': '#FEC7E4',  # 浅粉色
    'color6': '#E0FFFA',  # 浅青色
    'color7': '#FF9B9B',  # 浅红色
    'color8': '#44CA79'   # 深绿色
}

# 为不同实验场景预设颜色
PROTOCOL_COLORS = {
    'FDMMBF-ARP': PLOT_COLORS['color1'],
    'LEACH': PLOT_COLORS['color2'],
    'TEEN': PLOT_COLORS['color3'],
    'APTEEN': PLOT_COLORS['color4'],
    'HEED': PLOT_COLORS['color5'],
    'D2CRP': PLOT_COLORS['color6']
}

MAC_COLORS = {
    'TDMA': PLOT_COLORS['color1'],
    'CSMA_CA': PLOT_COLORS['color2'],
    'HYBRID': PLOT_COLORS['color3'],
    'FDMMBF-ARP': PLOT_COLORS['color4']
}

class BaseStation:
    def __init__(self):
        self.x = 0
        self.y = 0


class SensorNode:
    def __init__(self):
        # 位置和基本属性
        self.xd = 0  # 节点x坐标
        self.yd = 0  # 节点y坐标
        self.d = 0  # 节点到基站距离
        self.Rc = 0  # 通信距离
        self.temp_rand = 0  # 随机数
        self.type = 'N'  # 节点类型
        self.selected = 'N'  # 是否被选中
        self.power = 0  # 节点能量
        self.CH = 0  # 簇头节点
        self.flag = 1  # 节点是否存活
        # 网络拓扑相关
        self.N = []  # 邻居节点集
        self.Num_N = 0  # 邻居节点数
        self.FN = []  # 前邻节点集
        self.Num_FN = 0  # 前邻节点数
        self.CN = []  # 前簇头节点集
        self.Num_CN = 0  # 前簇头节点数
        self.num_join = 0  # 加入簇的节点数
        # 缓存和路由
        self.buffer_size = 1000  # 缓冲区大小
        self.available_buffer = 1000  # 可用缓冲区
        self.parent = None  # 父节点
        self.is_gateway = False  # 是否为网关节点
        self.cluster_members = []  # 簇成员列表
        self.layer = 0  # 节点所在层级
        self.weight = 0  # 节点权重
        self.upstream_nodes = []  # 上游节点列表
        self.downstream_nodes = []  # 下游节点列表
        self.routing_table = {}  # 路由表
        self.hop_count = float('inf')  # 到基站的跳数
        # MAC层和能量管理
        self.state = 'SLEEP'  # 节点状态: 'SLEEP', 'ACTIVE', 'LIGHT_WAKE'
        self.time_slot = -1  # TDMA时隙
        self.contention_window = CW_MIN  # CSMA/CA的竞争窗口
        self.backoff_timer = 0  # 退避计时器
        self.is_channel_busy = False  # 信道是否忙碌
        self.duty_cycle = DUTY_CYCLE_MAX  # 工作周期
        self.tdma_slot = None  # TDMA时隙
        self.last_active_time = 0  # 上次活动时间
        self.data_buffer = []  # 数据缓冲区
        self.packets_sent = 0  # 发送的数据包数量
        self.packets_received = 0  # 接收的数据包数量
        self.power_level = 0  # 发射功率等级
        self.spreading_factor = 7  # 扩频因子
        self.role = 1  # 节点角色等级
        self.task_activity = 0.5  # 通信任务活跃度
        # 2. 节点状态机扩展
        self.last_report_value = None
        self.last_report_time = 0
        self.last_value = None
        self.data_history = []
        self.aoi = 0  # Age of Information
        self.has_uploaded = False
        self.light_wake_start_time = 0  # 轻唤醒开始时间
        self.confirm_count = 0  # 收到的邻居确认数
        self.waiting_confirm = False  # 是否等待确认
        self.last_metadata_broadcast = -1  # 上次广播元数据的轮次
        self.pending_soft_wake = False  # 是否等待确认
        # 协议流程分阶段时延属性
        self.wakeup_delay = 0
        self.control_delay = 0
        self.tdma_queue_delay = 0
        self.routing_delay = 0


class D2CRPNode(SensorNode):
    def __init__(self):
        super().__init__()
        self.id = None
        self.xd = 0  # x坐标
        self.yd = 0  # y坐标
        self.power = 0  # 剩余能量
        self.init_power = 0  # 初始能量
        self.flag = 1  # 节点状态(1:活跃, 0:死亡)
        self.type = 'N'  # 节点类型(N:普通节点, C:簇头, 1-hop:一跳节点, 2-hop:二跳节点)
        self.CH = None  # 所属簇头
        self.cluster_members = []  # 簇成员
        self.one_hop = set()  # 一跳邻居
        self.two_hop = set()  # 二跳邻居
        self.distance_factor = 0  # 距离因子
        self.energy_factor = 0  # 能量因子
        self.next_hop = None  # 下一跳节点
        self.relay_node = None  # 中继节点(对二跳节点)
        self.tdma_slot = None  # TDMA时隙


class PathEntry:
    def __init__(self, path, E_min, hops, Q_avg, delay_sum=0):
        self.path = path          # 路径上的节点列表
        self.E_min = E_min        # 路径上的最小剩余能量
        self.hops = hops          # 路径跳数
        self.Q_avg = Q_avg        # 平均链路质量
        self.delay_sum = delay_sum # 累计时延


# 初始化参数
random.seed(2)
n = 400  # 节点总数
rmax = 3000  # 迭代次数
is_display = False  # 是否动态显示节点分簇情况

if is_display:
    n = 40  # 节点总数
    rmax = 200  # 迭代次数

xm = 100  # x轴范围
ym = 100  # y轴范围
sink = BaseStation()
sink.x = 50
sink.y = 125

# 能量参数
Eelec = 50e-9  # 电子能量消耗
Efs = 10e-12  # 自由空间传播能量消耗
Emp = 0.0013e-12  # 多路径衰减能量消耗
ED = 5e-9  # 数据融合能量消耗
E0 = 0.6  # 初始能量
Emin = 0.001  # 节点存活所需的最小能量
Rmax = 15  # 初始通信距离
d0 = 87  # 通信距离阈值
packetLength = 4000  # 数据包长度
ctrPacketLength = 100  # 控制包长度
sensing_energy = 0.00000005  # 感知数据的能量消耗
LEACH_P = 0.5

# improve_leach特有参数
min_sensor_nodes = int((xm * ym) / (np.pi * Rmax * Rmax))  # 最少的簇头数
participating_nodes = min_sensor_nodes  # 每轮参与簇头选举的节点数
p = participating_nodes / n      # 簇头选举概率
#p = 0.6
control_packet_duration = 0.001  # 控制帧持续时间(s)
total_broadcast_time = min_sensor_nodes * control_packet_duration  # 广播总时间

# 数据传输相关参数
DATA_RATE = 250000  # 数据传输速率 250kbps
AGGREGATION_RATIO = 0.6  # 数据聚合比例，表示聚合后的数据包数量与原始数据包数量的比值


def calculate_energy_consumption(node, packet_size, dist, is_control=False):
    """计算能量消耗，包括发送和接收能量"""
    if is_control:
        packet_length = ctrPacketLength
    else:
        packet_length = packetLength

    # 发送能量
    if dist > d0:
        ETx = packet_length * (Eelec + Emp * (dist ** 4))
    else:
        ETx = packet_length * (Eelec + Efs * (dist ** 2))
    
    # 接收能量
    ERx = packet_length * Eelec

def calculate_energy_consumption(node, packet_size, dist, is_control=False):
    """计算能量消耗，包括发送和接收能量"""
    if is_control:
        packet_length = ctrPacketLength
    else:
        packet_length = packetLength
        
    # 计算发送能量
    if dist > d0:
        ETx = Eelec * packet_length + Emp * packet_length * (dist ** 4)
    else:
        ETx = Eelec * packet_length + Efs * packet_length * (dist ** 2)
    
    # 计算接收能量
    ERx = Eelec * packet_length
    
    return ETx, ERx


# === 轻唤醒通信函数 ===
def broadcast_metadata(node, Node, r):
    # 广播元数据给所有邻居
    for neighbor_idx in node.N:
        Node[neighbor_idx].receive_metadata(node, r)


def receive_metadata(self, sender, r):
    # 邻居收到元数据后，若自己也检测到异常或同意唤醒，则反馈确认
    # 这里只做简单模拟：如果本节点本轮也触发软阈值，则反馈
    if hasattr(self, 'pending_soft_wake') and self.pending_soft_wake:
        self.receive_confirm(sender)


def receive_confirm(self, sender):
    self.confirm_count += 1


# 动态绑定到SensorNode类
def patch_sensor_node():
    SensorNode.receive_metadata = receive_metadata
    SensorNode.receive_confirm = receive_confirm


patch_sensor_node()


def insert_path_entry(path_list, new_entry, K):
    """
    将新路径插入到路径列表中，保持按优先级排序，只保留最优的K条
    优先级：剩余能量 > 跳数 > 链路质量
    """
    # 检查是否已有完全相同的路径
    for entry in path_list:
        if entry.path == new_entry.path:
            return False
    
    # 计算新路径的优先级分数
    def calculate_priority_score(entry):
        # 归一化各个指标
        energy_score = entry.E_min / 100.0  # 假设初始能量为100
        hop_score = 1.0 / (entry.hops + 1)  # 跳数越少分数越高
        quality_score = entry.Q_avg
        
        # 加权组合，能量权重最高
        return 0.1 * energy_score + 0.8 * hop_score + 0.1 * quality_score
    
    # 计算新路径的分数
    new_score = calculate_priority_score(new_entry)
    
    # 找到合适的插入位置
    insert_pos = 0
    for i, entry in enumerate(path_list):
        if calculate_priority_score(entry) < new_score:
            insert_pos = i
            break
        insert_pos = i + 1
    
    # 插入新路径
    path_list.insert(insert_pos, new_entry)
    
    # 只保留最优的K条路径
    if len(path_list) > K:
        path_list.pop()
    
    return True

def bellman_ford_multiobjective(gateway_nodes, sink, Node, K=3):
    """
    多目标Bellman-Ford算法，为每个网关节点找到K条到基站的最优路径
    考虑路径上的最小剩余能量、跳数和链路质量
    """
    # 初始化路径表
    path_tables = {node: [] for node in gateway_nodes}
    path_tables[sink] = [PathEntry([sink], sink.power, 0, 1.0)]  # 基站初始路径
    
    # 网关节点初始化
    for node in gateway_nodes:
        if node != sink:
            path_tables[node] = [PathEntry([node], node.power, 0, 1.0)]
    
    # 多轮松弛
    n = len(gateway_nodes) + 1  # 包含基站
    for _ in range(n-1):  # 最多n-1轮
        updated = False
        for u in gateway_nodes:
            for v in gateway_nodes + [sink]:
                if u != v:
                    # 检查是否在通信范围内
                    dist = calculate_distance(u, v)
                    if dist <= calculate_communication_radius(u):
                        # 获取链路质量
                        link_q = get_link_quality(u, v)
                        
                        # 尝试扩展所有已知路径
                        for path_entry in path_tables[v]:
                            if u not in path_entry.path:  # 避免环路
                                # 计算新路径的参数
                                new_path = [u] + path_entry.path
                                new_E_min = min(u.power, path_entry.E_min)
                                new_hops = path_entry.hops + 1
                                new_Q_avg = (path_entry.Q_avg * path_entry.hops + link_q) / new_hops
                                
                                # 创建新的路径条目
                                new_entry = PathEntry(new_path, new_E_min, new_hops, new_Q_avg)
                                
                                # 尝试将新路径插入到路径表中
                                if insert_path_entry(path_tables[u], new_entry, K):
                                    updated = True
        
        if not updated:  # 如果没有更新，提前结束
            break
    
    return path_tables

def select_route(path_list, strategy='main'):
    if not path_list:
        return None
    # 只返回最优路径（主路径）
    return path_list[0].path


def compute_data_importance(node, current_value, current_round, alpha=ALPHA, beta=BETA):
    if node.last_value is not None:
        diff = abs(current_value - node.last_value)
    else:
        diff = 0
    if node.data_history:
        mu = sum(node.data_history) / len(node.data_history)
    else:
        mu = current_value
    deviation = abs(current_value - mu)
    I = alpha * diff + beta * deviation
    return I


def calculate_power_level(node, target_node, E0, d0):
    """根据节点能量和目标节点距离计算发射功率等级"""
    # 发射功率范围
    Ptx_max = 0  # dBm
    Ptx_min = -25  # dBm

    # 距离和路径损耗（假设自由空间 d^2 或多径 d^4）
    d = calculate_distance(node, target_node)
    if d < d0:
        required_power = Efs * (d ** 2)
    else:
        required_power = Emp * (d ** 4)

    # 剩余能量比例控制权重（越少越节省能量）
    energy_ratio = max(0, min(1, node.power / E0))
    adaptive_power = Ptx_min + energy_ratio * (Ptx_max - Ptx_min)

    # 实际功率取 max(所需功率, 能量控制功率)
    Ptx = min(Ptx_max, max(adaptive_power, required_power))

    # 归一化发射功率等级
    return (Ptx - Ptx_min) / (Ptx_max - Ptx_min)


def calculate_spreading_factor(node):
    """计算节点的扩频因子"""
    SF_min = 7
    SF_max = 12
    SF = node.spreading_factor
    return (SF - SF_min) / (SF_max - SF_min)


def calculate_communication_radius(node):
    """计算节点的通信半径"""
    Rcbase = 10  # 基础通信半径

    # 计算各个参数
    Pi = calculate_power_level(node, sink, E0, d0)
    power_factor = 0.5 + Pi
    SFi = calculate_spreading_factor(node)
    sf_factor = 0.8 + 0.4 * SFi

    # 根据节点类型设置角色等级
    if node.type == 'N':
        Ri = 1  # 普通节点
    elif node.type == 'C':
        Ri = 2  # 簇头
    else:
        Ri = 3  # 网关节点

    # 设置任务活跃度
    if node.flag == 0:
        Ti = 0.2  # 无任务
    elif node.type == 'N':
        Ti = 0.5  # 准备状态
    else:
        Ti = 1.0  # 活跃阶段

    # 计算通信半径
    Rc = Rcbase * power_factor * sf_factor * Ri * Ti

    # 确保通信半径在合理范围内
    return max(Rcbase / 2, min(Rc, Rmax))


def calculate_throughput(packets_received, simulation_time):
    """计算吞吐量 (bps)"""
    return (packets_received * packetLength) / simulation_time


def calculate_delay(node, sink, packet_size, Node):
    """计算数据包从节点到基站的传输延时，增加实际网络环境的波动因素"""
    total_delay = 0

    def get_link_quality(node1, node2):
        distance = calculate_distance(node1, node2)
        # 增加随机干扰
        interference = random.uniform(0.1, 0.3)
        # 距离影响
        distance_factor = max(0.1, 1 - (distance / 800))
        # 链路质量波动
        link_quality = max(0.1, distance_factor - interference)
        return link_quality

    # 1. 普通节点到簇头的延时计算
    if node.type != 'C' and node.CH is not None:
        ch_node = Node[node.CH]
        dist_to_ch = calculate_distance(node, ch_node)
        link_quality = get_link_quality(node, ch_node)

        # 传播延时加入环境干扰
        propagation_delay = (dist_to_ch / 3e8) * (1 + random.uniform(0, 0.5))

        # 传输延时加入链路质量影响
        transmission_delay = packet_size / (250e3 * link_quality)

        # 处理延时加入负载波动
        cluster_load = sum(1 for n in Node if n.CH == node.CH)
        processing_delay = 0.001 * (1 + cluster_load / 5) * random.uniform(0.8, 1.5)

        # 使用正确的随机函数模拟队列延时
        queue_delay = -0.002 * math.log(1 - random.random()) * (1 + cluster_load / 10)

        # CSMA/CA竞争延时
        contention_window = random.randint(1, 32)  # 随机退避窗口
        contention_delay = random.uniform(0, contention_window * 0.000032)

        total_delay += (propagation_delay + transmission_delay +
                        processing_delay + queue_delay + contention_delay)

        node = ch_node

    # 2. 计算到基站的延时
    final_dist = calculate_distance(node, sink)
    final_link_quality = get_link_quality(node, sink)

    # 最后一跳延时计算
    final_propagation_delay = (final_dist / 3e8) * (1 + random.uniform(0, 0.3))
    final_transmission_delay = packet_size / (250e3 * final_link_quality)
    final_processing_delay = 0.001 * random.uniform(0.9, 1.3)

    # 重传机制
    retransmission_prob = 0.05  # 20%的重传概率
    if random.random() < retransmission_prob:
        # 重传会导致额外延时,在20%的概率下，发生重传。重传会增加额外的延时，延时值是传输和传播延时的加权平均。
        retransmission_delay = (final_transmission_delay + final_propagation_delay) * \
                               random.uniform(1.0, 1.5)
        total_delay += retransmission_delay

    total_delay += (final_propagation_delay + final_transmission_delay +
                    final_processing_delay)

    # 网络抖动
    jitter = random.gauss(0, 0.001)  # 均值为0，标准差为2ms
    total_delay += abs(jitter)

    # 突发延时
    if random.random() < 0.05:  # 15%的概率发生突发延时,模拟网络负载激增或临时故障
        burst_delay = random.uniform(0.005, 0.01)  # 5-20ms的突发延时
        total_delay += burst_delay

    return total_delay


def calculate_round_pdr(Node, sink, round_packets_sent):
    total_received = 0

    for i in range(n):
        if Node[i].flag != 0:
            if Node[i].type == 'C':  # 簇头节点
                # 计算簇头到下一跳或基站的PDR
                next_node = None
                if Node[i].is_gateway:
                    next_node = Node[i].parent if Node[i].parent else sink
                else:
                    next_node = sink

                # 计算链路质量
                link_quality = get_link_quality(Node[i], next_node)

                # 计算队列状态
                queue_status = len(Node[i].data_buffer) / MAX_BUFFER_SIZE

                # 计算能量状态
                energy_status = Node[i].power / E0

                # 计算网关路径质量（如果存在）
                gateway_factor = 1.0
                if Node[i].parent and Node[i].parent.is_gateway:
                    gateway_factor = get_link_quality(Node[i].parent, sink)

                # 计算簇内通信质量
                cluster_quality = 0
                if Node[i].cluster_members:
                    cluster_qualities = [get_link_quality(Node[i], Node[member])
                                         for member in Node[i].cluster_members]
                    cluster_quality = sum(cluster_qualities) / len(cluster_qualities)

                # 综合成功率计算
                success_rate = (link_quality * 0.25 +
                                (1 - queue_status) * 0.2 +
                                energy_status * 0.25 +
                                gateway_factor * 0.15 +
                                cluster_quality * 0.15)

                # 累加成功接收的数据包
                total_received += len(Node[i].cluster_members) * success_rate

            elif Node[i].CH != -1:  # 簇成员节点
                # 计算到簇头的链路质量
                ch_link_quality = get_link_quality(Node[i], Node[Node[i].CH])

                # 计算簇成员的能量状态
                member_energy_status = Node[i].power / E0

                # 计算簇成员的成功率
                member_success_rate = (ch_link_quality * 0.6 +
                                       member_energy_status * 0.4)

                # 累加成功接收的数据包
                total_received += member_success_rate

    return total_received / round_packets_sent if round_packets_sent > 0 else 0


def get_link_quality(node1, node2):
    distance = calculate_distance(node1, node2)
    # 基于距离的信号衰减
    path_loss = 20 * math.log10(distance) if distance > 0 else 0
    # 考虑环境噪声
    snr = MAX_SNR - path_loss
    # 归一化链路质量
    return max(0, min(1, snr / MAX_SNR))


def calculate_amrp(node, neighbors, Node):
    """计算平均最小可达功率(AMRP)"""
    if not neighbors:
        return float('inf')
    total_min_power = 0
    for neighbor in neighbors:
        dist = calculate_distance(node, Node[neighbor])
        # 计算与邻居通信所需的最小功率
        if dist <= d0:
            min_power = Eelec * packetLength + Efs * packetLength * (dist ** 2)
        else:
            min_power = Eelec * packetLength + Emp * packetLength * (dist ** 4)
        total_min_power += min_power
    return total_min_power / len(neighbors)


def calculate_heed_pdr(nodes, sink, round_packets_sent):
    total_received = 0

    for node in nodes:
        if node.flag != 0:  # 检查节点是否存活
            if node.type == 'C':  # 簇头节点
                # 计算邻居节点
                neighbors = [j for j in range(len(nodes)) if
                             nodes[j].flag != 0 and nodes.index(node) != j and
                             calculate_distance(node, nodes[j]) <= Rmax]

                # AMRP因子（HEED特有）
                amrp_factor = calculate_amrp(node, neighbors, nodes)
                normalized_amrp = min(1.0, 1 / amrp_factor) if amrp_factor > 0 else 0

                # 链路质量
                link_quality = get_link_quality(node, sink)

                # 队列状态
                queue_status = len(node.data_buffer) / MAX_BUFFER_SIZE

                # 能量状态
                energy_status = node.power / E0

                # 计算簇内通信质量
                cluster_quality = 0
                if hasattr(node, 'cluster_members') and node.cluster_members:
                    cluster_qualities = [get_link_quality(node, nodes[member])
                                         for member in node.cluster_members]
                    cluster_quality = sum(cluster_qualities) / len(cluster_qualities)

                # 综合成功率计算
                success_rate = (link_quality * 0.3 +
                                (1 - queue_status) * 0.2 +
                                energy_status * 0.3 +
                                cluster_quality * 0.2)

                # 考虑簇成员数量
                if hasattr(node, 'cluster_members'):
                    total_received += len(node.cluster_members) * success_rate
                else:
                    total_received += success_rate

            elif node.CH != -1:  # 簇成员节点
                # 计算到簇头的链路质量
                ch_node = nodes[node.CH]
                ch_link_quality = get_link_quality(node, ch_node)

                # 计算簇成员的能量状态
                member_energy_status = node.power / E0

                # 计算簇成员的成功率
                member_success_rate = (ch_link_quality * 0.6 +
                                       member_energy_status * 0.4)

                total_received += member_success_rate

    return total_received / round_packets_sent if round_packets_sent > 0 else 0


def calculate_leach_pdr(Node, sink, round_packets_sent):
    """改进的LEACH协议PDR计算"""
    total_received = 0

    for i in range(n):
        if Node[i].flag != 0:
            if Node[i].type == 'C':  # 簇头节点
                # 计算链路质量
                link_quality = get_link_quality(Node[i], sink)

                # 计算队列状态
                queue_status = len(Node[i].data_buffer) / MAX_BUFFER_SIZE

                # 计算能量状态
                energy_status = Node[i].power / E0

                # 计算簇内通信质量
                cluster_quality = 0
                if Node[i].cluster_members:
                    cluster_qualities = [get_link_quality(Node[i], Node[member])
                                         for member in Node[i].cluster_members]
                    cluster_quality = sum(cluster_qualities) / len(cluster_qualities)

                # 综合成功率计算
                success_rate = (link_quality * 0.3 +
                                (1 - queue_status) * 0.2 +
                                energy_status * 0.3 +
                                cluster_quality * 0.2)

                total_received += len(Node[i].cluster_members) * success_rate

            elif Node[i].CH != -1:  # 簇成员节点
                # 计算到簇头的链路质量
                ch_link_quality = get_link_quality(Node[i], Node[Node[i].CH])

                # 计算簇成员的能量状态
                member_energy_status = Node[i].power / E0

                # 计算簇成员的成功率
                member_success_rate = (ch_link_quality * 0.6 +
                                       member_energy_status * 0.4)

                total_received += member_success_rate

    return total_received / round_packets_sent if round_packets_sent > 0 else 0


def calculate_energy_std(nodes):
    """
    计算活跃节点能量的标准差
    参数:
        nodes: 节点列表或字典
    返回:
        float: 能量标准差
    """
    # 获取活跃节点的能量列表
    if isinstance(nodes, dict):
        alive_nodes = [node.power for node in nodes.values() if node.flag != 0 and node.power > 0]
    else:
        alive_nodes = [node.power for node in nodes if node.flag != 0 and node.power > 0]
    
    # 如果没有活跃节点或只有一个节点，返回nan
    if len(alive_nodes) <= 1:
        return np.nan
        
    # 计算平均能量
    mean_energy = np.mean(alive_nodes)
    
    # 计算标准差
    std_dev = np.std(alive_nodes)
    
    # 如果标准差太小（接近0），返回一个最小值而不是0
    if std_dev < 1e-10:
        return 1e-10
        
    return float(std_dev)  # 确保返回Python float类型


def calculate_edge_weight(gateway1, gateway2, sink):
    """综合考虑 物理距离、节点剩余能量 和 可用缓冲区大小，来计算两个网关节点之间的边权重。
    路径的边权重越大，表示这条路径的传输成本越高，可能需要更多的能量或者会受到负载的限制，
    导致选择这样的路径时需要额外的代价。"""
    distance = calculate_distance(gateway1, gateway2)
    energy_factor = min(gateway1.power, gateway2.power) / E0  # 计算能量因子，根据两个网关节点的剩余能量来确定
    load_factor = min(gateway1.available_buffer, gateway2.available_buffer) / 1000  # 计算负载因子，基于两个网关节点的可用缓冲区来确定
    weight = distance * (1 / energy_factor) * (1 / load_factor)
    return weight


def bellman_ford(gateway_nodes, sink):
    n_gateways = len(gateway_nodes)
    distance = {node: float('inf') for node in gateway_nodes}
    predecessor = {node: None for node in gateway_nodes}
    virtual_sink = SensorNode()
    virtual_sink.xd = sink.x
    virtual_sink.yd = sink.y
    virtual_sink.power = float('inf')
    virtual_sink.available_buffer = float('inf')
    gateway_nodes.append(virtual_sink)
    distance[virtual_sink] = 0
    for _ in range(n_gateways - 1):
        for u in gateway_nodes:
            for v in gateway_nodes:
                if u != v:
                    weight = calculate_edge_weight(u, v, sink)
                    if distance[u] + weight < distance[v]:
                        distance[v] = distance[u] + weight
                        predecessor[v] = u

    gateway_nodes.remove(virtual_sink)
    return distance, predecessor


def forward_through_gateways(gateway_nodes, sink, Node, K=3):
    """通过网关转发数据到基站，支持多路径并行转发和智能异常检测"""
    if not gateway_nodes:
        return

    path_tables = bellman_ford_multiobjective(gateway_nodes, sink, Node, K)

    for gateway in gateway_nodes:
        if gateway.power <= Emin:
            continue
            
        # 获取当前路径的时延
        current_delay = calculate_delay(gateway, sink, packetLength, Node)
        
        # 检查是否需要重路由
        if hasattr(gateway, 'current_path') and detect_delay_anomaly(gateway, current_delay):
            # 执行自适应局部重路由
            if adaptive_local_reroute(gateway, sink, Node):
                # 更新路径表中的时延信息
                for path_entry in path_tables[gateway]:
                    if path_entry.path == gateway.current_path.path:
                        path_entry.delay_history.append(current_delay)
                        path_entry.delay_variance = np.var(path_entry.delay_history) if len(path_entry.delay_history) > 1 else 0
        
        # 使用并行多路径转发
        parallel_forward(gateway, path_tables, sink, Node)
        
        # 更新路径的时延历史
        if path_tables[gateway]:
            main_path = path_tables[gateway][0]
            if not hasattr(gateway, 'current_path'):
                gateway.current_path = main_path
            main_path.delay_history.append(current_delay)
            main_path.delay_variance = np.var(main_path.delay_history) if len(main_path.delay_history) > 1 else 0


def perform_csma_ca(node, max_backoff=3, failure_strategy='switch'):
    """
    执行CSMA/CA,支持可配置的最大重传次数和失败处理策略
    Args:
        node: 节点对象
        max_backoff: 最大重传次数,默认为3
        failure_strategy: 失败处理策略 ('switch' 或 'drop')
    Returns:
        tuple: (是否传输成功, 失败处理建议)
    """
    backoff_count = 0
    while backoff_count < max_backoff:
        # 执行退避
        cw = min(pow(2, backoff_count) * CW_MIN, CW_MAX) 
        backoff_slots = random.randint(0, cw-1)
        
        # 模拟信道检测
        channel_busy = random.random() < 0.3  # 30%概率信道忙
        if not channel_busy:
            return True, None
        
        backoff_count += 1
        time.sleep(SLOT_TIME * backoff_slots)  # 模拟退避时延
    
    # 达到最大退避次数后的处理
    if failure_strategy == 'switch':
        return False, 'switch_path'  # 建议切换路径
    else:  # 'drop'策略
        return False, 'drop_packet'  # 建议丢弃数据包

def run_backoff_comparison():
    """
    对比不同重传次数(3-7)对网络性能的影响
    """
    print("开始退避重传次数对比实验...")

    global n, rmax
    n = 100   
    rounds = 3  
    
    # 2. 关闭图形显示
    global is_display
    is_display = True
    
    # 3. 添加简单的传感器数据模拟
    def simulate_sensor_data(node, r):
        return random.uniform(0, 1)
    
    # 4. 其他代码保持不变
    backoff_attempts = range(3, 8)
    results = {
        'delays': [],
        'pdrs': []
    }
    
    for max_backoff in backoff_attempts:
        print(f"\n测试重传次数: {max_backoff}")
        delays = []
        pdrs = []
        
        try:
            Node = initialize_network()
            for r in range(rounds):
                print(f"\n当前进度: {r+1}/{rounds}")
                
                global MAX_BACKOFF
                MAX_BACKOFF = max_backoff
                
                result = run_improve_leach(
                    config={
                        'max_backoff': max_backoff,
                        'num_nodes': n,
                        'round': r
                    }
                )
                
                if result:
                    _, _, pdr, delay, _, _, _, _, _ = result
                    delays.append(np.mean(delay[delay > 0]))
                    pdrs.append(np.mean(pdr[pdr > 0]))
            
            print(f"\n完成重传次数{max_backoff}的测试")
            if delays and pdrs:
                avg_delay = np.mean(delays)
                avg_pdr = np.mean(pdrs)
                results['delays'].append(avg_delay)
                results['pdrs'].append(avg_pdr)
                print(f"平均时延: {avg_delay:.4f}")
                print(f"平均PDR: {avg_pdr:.4f}")
            
        except Exception as e:
            print(f"\n测试出错: {str(e)}")
            continue
    
    if results['delays'] and results['pdrs']:
        plot_backoff_comparison(backoff_attempts, results)
        save_backoff_results(backoff_attempts, results)
    
    return results  # 添加这行，返回结果字典

def plot_backoff_comparison(backoff_attempts, results):
    """
    绘制重传次数对比图表
    """
    plt.figure(figsize=(12, 6))
    
    # 设置双Y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 绘制时延曲线
    line1 = ax1.plot(backoff_attempts, results['delays'], color='#A4C2F4', 
                     label='时延', marker='o', markersize=8, linewidth=2)
    ax1.set_xlabel('最大重传次数')
    ax1.set_ylabel('平均时延(ms)')
    
    # 绘制PDR曲线
    line2 = ax2.plot(backoff_attempts, results['pdrs'], color='#B6D7A8', 
                     label='PDR', marker='s', markersize=8, linewidth=2)
    ax2.set_ylabel('PDR')
    
    # 设置图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper right')
    
    plt.title('不同重传次数的时延和PDR对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴刻度
    plt.xticks(backoff_attempts)
    
    plt.tight_layout()
    plt.savefig('backoff_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_backoff_results(backoff_attempts, results):
    """
    保存实验结果到表格
    """
    # 创建结果表格
    plt.figure(figsize=(8, 6))
    table_data = []
    headers = ['组合编号', '最大重传次数', '平均时延(ms)', 'PDR']
    
    for i, max_backoff in enumerate(backoff_attempts, 1):
        table_data.append([
            i,
            max_backoff,
            f"{results['delays'][i-1]:.4f}",
            f"{results['pdrs'][i-1]:.4f}"
        ])
    
    # 创建表格
    table = plt.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#FFF2CC'] * len(headers),
        cellColours=[['#FFF2CC' if i % 2 == 0 else '#D9EAD3' 
                     for _ in range(len(headers))] 
                    for i in range(len(table_data))]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 隐藏坐标轴
    plt.axis('off')
    
    plt.title('重传次数实验结果表')
    plt.tight_layout()
    
    # 保存表格
    plt.savefig('backoff_results_table.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

def assign_tdma_slot(cluster_head, cluster_members):
    # 为簇成员分配TDMA时隙
    if not cluster_members:
        return {}
    slot_duration = FRAME_TIME / len(cluster_members)
    time_slots = {}
    for idx, member in enumerate(cluster_members):
        time_slots[member] = {
            'start': idx * slot_duration,
            'end': (idx + 1) * slot_duration
        }
    return time_slots


def calculate_network_load(nodes):
    # 计算网络负载
    total_buffer = sum(len(node.data_buffer) for node in nodes if node.flag != 0)
    max_buffer = sum(node.buffer_size for node in nodes if node.flag != 0)
    return total_buffer / max_buffer if max_buffer > 0 else 0


def updata_duty_cycle(node, network_load):
    # 更新设备的工作周期
    if network_load < LOAD_THRESHOLD_LOW:
        node.duty_cycle = DUTY_CYCLE_MIN
    elif network_load > LOAD_THRESHOLD_HIGH:
        node.duty_cycle = DUTY_CYCLE_MAX
    else:
        # 线性调整工作周期
        node.duty_cycle = DUTY_CYCLE_MIN + (DUTY_CYCLE_MAX - DUTY_CYCLE_MIN) * network_load


def calculate_node_layer(node, sink):
    """计算节点所在层级"""
    dist_to_sink = calculate_distance(node, sink)
    return min(int(dist_to_sink / Rmax), int((xm * ym) ** 0.5 / Rmax))


def calculate_routing_weight(current_node, candidate_node, sink):
    """计算路由权重"""
    dist_weight = 1 - (calculate_distance(current_node, candidate_node) / Rmax)
    energy_weight = candidate_node.power / E0
    buffer_weight = candidate_node.available_buffer / candidate_node.buffer_size
    hop_weight = 1 / (candidate_node.hop_count + 1)
    return 0.3 * dist_weight + 0.3 * energy_weight + 0.2 * buffer_weight + 0.2 * hop_weight

def calculate_gateway_weight(node_i, node_j, gateway, sink, Node):
    """计算网关节点权重"""
    dist_i_sink = calculate_distance(node_i, sink)
    dist_j_sink = calculate_distance(node_j, sink)
    dw1 = (dist_i_sink - dist_j_sink) / dist_i_sink
    dist_i_j = calculate_distance(node_i, node_j)
    dist_j_gateway = calculate_distance(node_j, gateway)
    dist_gateway_sink = calculate_distance(gateway, sink)
    if abs(dist_j_gateway + dist_gateway_sink - dist_j_sink) < 0.1 * dist_j_sink:
        dw2 = (2 * Rmax - (dist_j_gateway + dist_gateway_sink)) / (2 * Rmax - dist_j_sink)
    else:
        dw2 = (2 * Rmax - (dist_i_sink + dist_j_sink)) / (2 * Rmax - dist_i_j)
    max_neighbor_energy = max(Node[j].power for j in gateway.N) if gateway.N else gateway.power
    dw3 = gateway.power / max_neighbor_energy
    dw4 = gateway.available_buffer / gateway.buffer_size
    return dw1 * dw2 * dw3 * dw4

def calculate_parent_weight(current_node, candidate_parent, sink, is_gateway_parent, Node):
    """计算父节点选择权重"""
    max_neighbor_energy = max(
        Node[j].power for j in candidate_parent.N) if candidate_parent.N else candidate_parent.power
    dw3 = candidate_parent.power / max_neighbor_energy
    
    dw4 = candidate_parent.available_buffer / candidate_parent.buffer_size
    dist_parent_sink = calculate_distance(candidate_parent, sink)
    dist_current_sink = calculate_distance(current_node, sink)
    dw5 = (dist_current_sink - dist_parent_sink) / Rmax if is_gateway_parent else (
            dist_parent_sink - dist_current_sink) / Rmax
            
    return dw3 * dw4 * dw5

def build_routing_table(nodes, sink):
    """构建路由表"""
    # 初始化
    for node in nodes:
        if node.flag != 0:
            node.hop_count = float('inf')
            node.routing_table.clear()

    # 设置基站相邻节点
    for i, node in enumerate(nodes):
        if node.flag != 0 and calculate_distance(node, sink) <= Rmax:
            node.hop_count = 1
            node.routing_table[sink] = {'next_hop': None, 'weight': 1.0}

    # 迭代构建
    changed = True
    while changed:
        changed = False
        for i, current_node in enumerate(nodes):
            if current_node.flag == 0:
                continue

            for j, neighbor_node in enumerate(nodes):
                if i == j or neighbor_node.flag == 0:
                    continue

                if calculate_distance(current_node, neighbor_node) <= Rmax:
                    if neighbor_node.hop_count + 1 < current_node.hop_count:
                        current_node.hop_count = neighbor_node.hop_count + 1
                        weight = calculate_routing_weight(current_node, neighbor_node, sink)
                        current_node.routing_table[sink] = {
                            'next_hop': j,
                            'weight': weight
                        }
                        changed = True


def select_gateway_nodes(nodes, current_layer, sink, weight_config='original'):
    """选择网关节点"""
    gateway_nodes = []
    layer_nodes = [node for node in nodes if node.flag != 0 and node.layer == current_layer]
    weights = {
        'original': {'w1': 0.4, 'w2': 0.3, 'w3': 0.3},
        'distance': {'w1': 0.3, 'w2': 0.4, 'w3': 0.3},
        'energy': {'w1': 0.3, 'w2': 0.3, 'w3': 0.4},
        'buffer': {'w1': 0.6, 'w2': 0.2, 'w3': 0.2},
        'hybrid': {'w1': 0.5, 'w2': 0.3, 'w3': 0.2},
        'fuzzy': {'w1': 0.4, 'w2': 0.4, 'w3': 0.2}
    }
    w = weights[weight_config]
    for node in layer_nodes:
        if node.power > 0.4 * E0 and node.available_buffer > 0.3 * node.buffer_size:
            upper_connections = sum(1 for n in nodes if n.flag != 0 and
                                    n.layer == current_layer - 1 and
                                    calculate_distance(node, n) <= Rmax)
            lower_connections = sum(1 for n in nodes if n.flag != 0 and
                                    n.layer == current_layer + 1 and
                                    calculate_distance(node, n) <= Rmax)

            if upper_connections > 0 and lower_connections > 0:
                weight = (node.power / E0 * w['w1'] +
                          (node.available_buffer / node.buffer_size) * w['w2'] +
                          (upper_connections + lower_connections) / (2 * n) * w['w3'])
                node.gateway_weight = weight
                if weight > 0.7:
                    node.is_gateway = True
                    gateway_nodes.append(node)
    return gateway_nodes


def calculate_distance(node1, node2):
    """计算两个节点间的距离"""
    if isinstance(node2, BaseStation):
        return ((node1.xd - node2.x) ** 2 + (node1.yd - node2.y) ** 2) ** 0.5
    else:
        return ((node1.xd - node2.xd) ** 2 + (node1.yd - node2.yd) ** 2) ** 0.5


def create_fuzzy_system():
    """创建模糊逻辑系统"""
    # 定义输入变量
    energy = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'energy')
    neighbor_density = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'neighbor_density')
    distance = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'distance')
    
    # 定义输出变量
    chance = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'chance')
    
    # 为输入变量定义模糊集
    energy['low'] = fuzz.trimf(energy.universe, [0, 0, 0.4])
    energy['medium'] = fuzz.trimf(energy.universe, [0.2, 0.5, 0.8])
    energy['high'] = fuzz.trimf(energy.universe, [0.6, 1, 1])
    
    neighbor_density['sparse'] = fuzz.trimf(neighbor_density.universe, [0, 0, 0.4])
    neighbor_density['moderate'] = fuzz.trimf(neighbor_density.universe, [0.2, 0.5, 0.8])
    neighbor_density['dense'] = fuzz.trimf(neighbor_density.universe, [0.6, 1, 1])
    
    distance['near'] = fuzz.trimf(distance.universe, [0, 0, 0.4])
    distance['medium'] = fuzz.trimf(distance.universe, [0.2, 0.5, 0.8])
    distance['far'] = fuzz.trimf(distance.universe, [0.6, 1, 1])
    
    # 为输出变量定义模糊集
    chance['very_low'] = fuzz.trimf(chance.universe, [0, 0, 0.25])
    chance['low'] = fuzz.trimf(chance.universe, [0, 0.25, 0.5])
    chance['medium'] = fuzz.trimf(chance.universe, [0.25, 0.5, 0.75])
    chance['high'] = fuzz.trimf(chance.universe, [0.5, 0.75, 1])
    chance['very_high'] = fuzz.trimf(chance.universe, [0.75, 1, 1])
    
    # 定义模糊规则
    rule1 = ctrl.Rule(energy['high'] & neighbor_density['moderate'] & distance['near'], chance['very_high'])
    rule2 = ctrl.Rule(energy['high'] & neighbor_density['dense'] & distance['near'], chance['high'])
    rule3 = ctrl.Rule(energy['medium'] & neighbor_density['moderate'] & distance['medium'], chance['medium'])
    rule4 = ctrl.Rule(energy['low'] | distance['far'], chance['very_low'])
    rule5 = ctrl.Rule(energy['medium'] & neighbor_density['sparse'], chance['low'])
    rule6 = ctrl.Rule(energy['high'] & distance['medium'] & neighbor_density['moderate'], chance['high'])
    rule7 = ctrl.Rule(energy['medium'] & distance['near'] & neighbor_density['dense'], chance['medium'])
    rule8 = ctrl.Rule(energy['low'] & neighbor_density['sparse'], chance['very_low'])
    rule9 = ctrl.Rule(energy['high'] & distance['far'] & neighbor_density['dense'], chance['medium'])
    
    # 创建控制系统
    ch_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    return ctrl.ControlSystemSimulation(ch_ctrl)

def calculate_fuzzy_chance(node, nodes, max_energy):
    """计算模糊机会值，用于簇头选举"""
    # 1. 能量因子计算
    # 获取邻居节点中的最大能量值，如果没有邻居则使用节点自身能量
    neighbor_max_energy = max(nodes[j].power for j in node.N) if node.N else node.power
    # 计算能量归一化值：节点能量/邻居最大能量
    norm_energy = node.power / neighbor_max_energy

    # 2. 邻居节点因子计算
    # 计算理想邻居数量：总节点数/最小传感器节点数
    optimal_neighbors = n / min_sensor_nodes
    # 获取当前邻居数量
    current_neighbors = len(node.N)
    # 归一化邻居数量：使用取模运算避免过大值
    norm_neighbors = (current_neighbors % (optimal_neighbors + 1)) / optimal_neighbors

    # 3. 距离因子计算
    if node.N:
        # 计算到所有邻居的平均距离
        avg_dist = (1 / norm_neighbors) * sum(
            calculate_distance(node, nodes[j]) for j in node.N
        )
        # 归一化距离：使用最大通信半径归一化
        norm_dist = (Rmax - avg_dist) / Rmax
    else:
        # 没有邻居时距离因子为0
        norm_dist = 0
    # 4. 综合计算模糊机会值
    # 能量权重50%，邻居数量权重30%，距离权重20%
    

    try:
        # 创建并配置模糊系统
        fuzzy_system = create_fuzzy_system()
        
        # 将预处理后的输入值输入到模糊系统
        fuzzy_system.input['energy'] = norm_energy
        fuzzy_system.input['neighbor_density'] = norm_neighbors
        fuzzy_system.input['distance'] = norm_dist
        
        # 执行模糊推理
        fuzzy_system.compute()
        fuzzy_chance = fuzzy_system.output['chance']
        return fuzzy_chance
    except Exception as e:
        # 异常时返回归一化输入的均值
        return np.mean([norm_energy, norm_neighbors, norm_dist])

def fuzzy_cluster_head_election(Node, r):
    """基于模糊逻辑的簇头选举过程"""
    candidate_CH = []
    max_energy = max(node.power for node in Node if node.flag != 0)
    
    # 第一阶段：通过LEACH阈值初筛
    potential_CH = []
    for i in range(len(Node)):
        if Node[i].flag != 0 and not Node[i].is_gateway:
            threshold = (p) / (1 - p * (r % round(1 / p)))
            if Node[i].temp_rand <= threshold:
                potential_CH.append(i)
    
    # 第二阶段：对初筛通过的节点使用模糊逻辑评估
    if potential_CH:
        chance_values = {}
        for i in potential_CH:
            chance_values[i] = calculate_fuzzy_chance(Node[i], Node, max_energy)
        
        # 根据网络规模动态确定最佳簇头数量
        active_nodes = sum(1 for node in Node if node.flag != 0)
        optimal_ch_count = max(int(active_nodes * 0.1), 1)  # 确保至少有一个簇头
        
        # 选择机会值最高的节点作为簇头
        selected_CH = sorted(chance_values.items(), key=lambda x: x[1], reverse=True)[:optimal_ch_count]
        
        for ch_id, ch_chance in selected_CH:
            # 执行CSMA/CA避免冲突
            success, action = perform_csma_ca(Node[ch_id])
            if success:
                Node[ch_id].type = 'C'
                Node[ch_id].selected = 'O'
                Node[ch_id].CH = -1
                Node[ch_id].Rc = calculate_communication_radius(Node[ch_id])
                candidate_CH.append(ch_id)
                
                # 广播簇头状态的能量消耗
                ETx, _ = calculate_energy_consumption(
                    Node[ch_id],
                    ctrPacketLength,
                    Node[ch_id].Rc,
                    is_control=True
                )
                Node[ch_id].power -= ETx
    
    return candidate_CH


def initialize_network():
    """初始化网络"""
    Node = []
    fig1 = plt.figure(dpi=80)
    plt.grid(linestyle="dotted")
    plt.scatter(sink.x, sink.y, marker='*', s=200)

    for i in range(n):
        node = SensorNode()
        node.xd = random.random() * xm
        node.yd = random.random() * ym
        node.d = calculate_distance(node, sink)
        node.Rc = calculate_communication_radius(node)
        node.temp_rand = random.random()
        node.power = E0
        Node.append(node)
        plt.scatter(node.xd, node.yd, marker='o')

    plt.legend(['基站', '节点'])
    plt.xlabel('x', fontdict={"family": "SimHei", "size": 15})
    plt.ylabel('y', fontdict={"family": "SimHei", "size": 15})
    plt.show()
    return Node, sink

    
def run_improve_leach(config=None):
    
    # 设置失败处理策略
    if config is None:
        config = {}
    failure_strategy = config.get('failure_strategy', 'switch')  # 默认使用切换策略
    max_backoff_failures = config.get('max_backoff_failures', 3)
    # 获取权重配置，默认为 'original'
    current_weight_config = config.get('weight_config', 'original')
    
    Node, _ = initialize_network()  # 使用_忽略返回的sink，因为我们已经有全局的sink了
    if is_display:
        plt.ion()  # Turn on interactive mode
        plt.figure()
    alive_improve_leach = np.zeros(rmax)
    re_improve_leach = np.zeros(rmax)
    pdr_improve_leach = np.zeros(rmax)
    delay_improve_leach = np.zeros(rmax)
    throughput_improve_leach = np.zeros(rmax)

    temp_delays = []
    window_size = 200
    delay_index = 0  # 用于记录delay_improve_leach的索引
    total_packets_sent = 0
    total_packets_received = 0
    # 在函数开始处添加标准差记录列表
    # 初始化吞吐量列表
    energy_std_data = []
    throughput_per_round = []
    round_time = 1  # 假设每轮持续1秒

    false_wakeup_count_list = []  # 每轮误唤醒节点数
    false_wakeup_total = 0  # 累计误唤醒次数
    false_wakeup_energy = 0  # 累计误唤醒能耗

    for r in range(rmax):
        # 更新统计信息
        false_wakeup_count = 0
        if r % 200 == 0:
            round_packets_sent = 0
            round_packets_received = 0
        
        # 计算当前轮次的能量标准差
        current_std = calculate_energy_std(Node)
        energy_std_data.append(current_std)
        
        for i in range(n):
            if Node[i].flag != 0:
                re_improve_leach[r] += Node[i].power
                alive_improve_leach[r] += 1

        if alive_improve_leach[r] == 0:
            # 若节点全死，后续补nan直到rmax
            for _ in range(r+1, rmax):
                energy_std_data.append(np.nan)
            break

        # 重置节点状态
        for i in range(n):
            Node[i].type = 'N'
            Node[i].selected = 'N'
            Node[i].temp_rand = random.random()
            Node[i].Rc = calculate_communication_radius(Node[i])
            Node[i].CH = 0
            Node[i].is_gateway = False
            Node[i].parent = None
            Node[i].cluster_members = []
            Node[i].layer = calculate_node_layer(Node[i], sink)
            Node[i].data_buffer = []
            Node[i].has_uploaded = False

            # 邻居节点
            Node[i].N = []
            for j in range(n):
                if i != j and Node[j].flag != 0:
                    dist = calculate_distance(Node[i], Node[j])
                    if dist < Node[i].Rc:
                        Node[i].N.append(j)
            Node[i].Num_N = len(Node[i].N)

        # 构建路由表
        build_routing_table(Node, sink)

        # 按层处理节点
        max_layer = max(node.layer for node in Node if node.flag != 0)
        for layer in range(max_layer, -1, -1):  # 从最远层向基站层遍历
            # 1. 选择网关节点
            gateway_nodes = select_gateway_nodes(Node, layer, sink, weight_config=current_weight_config)

            # 2. 簇头选举
            candidate_CH = []
            potential_CH = []  # 存储潜在的簇头候选

            # 第一阶段：通过阈值筛选潜在候选
            for i in range(n):
                if Node[i].flag != 0 and Node[i].layer == layer and not Node[i].is_gateway:
                    threshold = (p) / (1 - p * (r % round(1 / p)))
                    if Node[i].temp_rand <= threshold:
                        potential_CH.append(i)

            # 第二阶段：应用模糊逻辑进行决策
            if potential_CH:
                # 计算所有潜在簇头的模糊机会值
                max_energy = max(Node[i].power for i in potential_CH)
                chance_values = {}
                for i in potential_CH:
                    # 考虑多种因素计算模糊机会值
                    chance_values[i] = calculate_fuzzy_chance(Node[i], Node, max_energy)

                # 根据层级大小设置最佳簇头数量
                optimal_ch_count = max(int(len([n for n in Node if n.flag != 0 and n.layer == layer]) * 0.1), 1)

                # 按模糊机会值从高到低排序并选择最优的几个节点作为簇头
                selected_CH = sorted(chance_values.items(), key=lambda x: x[1], reverse=True)[:optimal_ch_count]
                for ch_id, ch_chance in selected_CH:
                    # 执行CSMA/CA避免冲突
                    success, action = perform_csma_ca(Node[ch_id], failure_strategy=failure_strategy)
                    if success:
                        Node[ch_id].type = 'C'
                        Node[ch_id].selected = 'O'
                        Node[ch_id].CH = -1
                        Node[ch_id].Rc = Rmax * Node[ch_id].power / E0
                        candidate_CH.append(ch_id)
                        # 广播簇头状态的能量消耗
                        ETx, _ = calculate_energy_consumption(
                            Node[ch_id],
                            ctrPacketLength,
                            Node[ch_id].Rc,
                            is_control=True
                        )
                        Node[ch_id].power -= ETx  # 只考虑发送能量
                    elif action == 'switch_path':
                        Node[ch_id].type = 'N'
                        Node[ch_id].CH = -2  # 标记为直连基站
                    elif action == 'connect_sink':
                        Node[ch_id].type = 'N'
                        Node[ch_id].CH = -2  # 标记为直连基站
                    elif action == 'drop_packet':
                        pass  # 什么都不做

            # 3. 簇成员加入(使用CSMA/CA)
            for i in range(n):
                if Node[i].flag != 0 and Node[i].type == 'N':
                    min_dist = float('inf')
                    selected_CH = -1
                    # 选择最佳簇头
                    for ch in candidate_CH:
                        dist = calculate_distance(Node[i], Node[ch])
                        if dist < Node[ch].Rc and dist < min_dist:
                            min_dist = dist
                            selected_CH = ch

                    if selected_CH != -1:  # 使用CSMA/CA进行加入请求
                        success, action = perform_csma_ca(Node[i], failure_strategy=failure_strategy)
                        if success:
                            # 加入簇的能量消耗
                            ETx, ERx = calculate_energy_consumption(
                                Node[i],
                                ctrPacketLength,
                                min_dist,
                                is_control=True
                            )
                            Node[i].power -= ETx  # 节点发送加入请求的能量
                            Node[selected_CH].power -= ERx  # 簇头接收加入请求的能量

                            Node[i].CH = selected_CH
                            Node[selected_CH].cluster_members.append(i)
                        elif action == 'switch_path':
                            Node[i].CH = -2  # 标记为直连基站
                        elif action == 'connect_sink':
                            continue
                        elif action == 'drop_packet':
                            continue  # 直接丢弃，不做任何处理
        # 4. 节点状态机调度（基于数据重要性和阈值）
        for i in range(n):
            node = Node[i]
            if node.flag == 0:
                continue
            # 休眠→感知
            previous_state = node.state
            if node.state == 'SLEEP':
                node.state = 'SENSING'
            # 感知→判断重要性
            if node.state == 'SENSING':
                value = simulate_sensor_data(node, r)
                I = compute_data_importance(node, value, r)
                node.data_history.append(value)
                node.last_value = value
                node.current_value = value

                # 动态能量自适应阈值
                energy_ratio = node.power / E0
                hard_th = HARD_THRESHOLD + 0.1 * (0.5 - energy_ratio)  # 降低硬阈值的调整幅度
                soft_th = SOFT_THRESHOLD + 0.05 * (0.5 - energy_ratio)  # 降低软阈值的调整幅度

                node.current_hard_th = hard_th
                node.current_soft_th = soft_th
                node.wakeup_reason = None
                node.pending_soft_wake = False
                # 硬阈值
                if I >= hard_th:
                    node.state = 'ACTIVE'
                    node.last_report_value = value
                    node.last_report_time = r
                    node.wakeup_reason = 'HARD_THRESHOLD'
                # 软阈值
                elif node.last_report_value is not None and abs(value - node.last_report_value) >= soft_th:
                    # 进入轻唤醒
                    node.state = 'LIGHT_WAKE'
                    node.light_wake_start_time = r
                    node.confirm_count = 0
                    node.waiting_confirm = True
                    node.pending_soft_wake = True
                    if node.last_metadata_broadcast != r:
                        broadcast_metadata(node, Node, r)
                        node.last_metadata_broadcast = r
                # 最大等待时间
                elif node.last_report_time is not None and (r - node.last_report_time) >= MAX_WAIT:
                    node.state = 'ACTIVE'
                    node.last_report_value = value
                    node.last_report_time = r
                    node.wakeup_reason = 'MAX_WAIT'
                else:
                    node.state = 'SLEEP'
            # 轻唤醒状态
            if node.state == 'LIGHT_WAKE':
                # 在确认窗口内统计邻居反馈
                if node.waiting_confirm and (r - node.light_wake_start_time) < 1:  # 这里用1轮模拟时间窗口
                    if node.confirm_count >= CONFIRM_THRESHOLD:
                        node.state = 'ACTIVE'
                        node.waiting_confirm = False
                        node.wakeup_reason = 'SOFT_THRESHOLD_CONFIRMED'
                    # 否则继续等待
                else:
                    # 超时未达确认数，回到SLEEP
                    node.state = 'SLEEP'
                    node.waiting_confirm = False
            # IDLE状态（可监听邻居广播，或等待下次感知）
            if node.state == 'IDLE':
                pass
            # ACTIVE状态：发送数据
            if node.state == 'ACTIVE':
                # 判断是否为误唤醒：如果是通过软阈值或最大等待时间唤醒，且当前值低于硬阈值
                if node.wakeup_reason in ['SOFT_THRESHOLD', 'SOFT_THRESHOLD_CONFIRMED',
                                          'MAX_WAIT'] and node.current_value < node.current_hard_th:
                    false_wakeup_count += 1
                    false_wakeup_total += 1
                    false_wakeup_energy += WAKEUP_POWER

                # 使用CSMA/CA发送数据到簇头/中继
                if node.type != 'C' and node.CH is not None and perform_csma_ca(node)[0]:
                    ch_node = Node[node.CH]
                    dist = calculate_distance(node, ch_node)
                    ETx, ERx = calculate_energy_consumption(node, packetLength, dist)
                    node.power -= ETx  # 发送节点消耗
                    ch_node.power -= (ERx + ED * packetLength)  # 簇头接收并融合数据
                node.state = 'SLEEP'

        false_wakeup_count_list.append(false_wakeup_count)

        # 5. 通过网关转发数据（使用CSMA/CA）
        for i in range(n):
            if Node[i].flag != 0 and Node[i].type == 'C' and perform_csma_ca(Node[i])[0]:
                best_gateway = None
                best_weight = -float('inf')
                for gateway in gateway_nodes:
                    if gateway.layer < Node[i].layer:
                        for j in Node[i].N:
                            weight = calculate_gateway_weight(Node[i], Node[j], gateway, sink, Node)
                            if weight > best_weight:
                                best_weight = weight
                                best_gateway = gateway
                if best_gateway:
                    ch_to_gateway_dist = calculate_distance(Node[i], best_gateway)
                    ETx, ERx = calculate_energy_consumption(Node[i], packetLength, ch_to_gateway_dist)
                    Node[i].power -= ETx  # 簇头发送消耗
                    best_gateway.power -= ERx  # 网关接收消耗
                    active_gateways = [g for g in gateway_nodes if g.power > Emin]
                    forward_through_gateways(active_gateways, sink, Node)
                else:
                    direct_to_sink_dist = calculate_distance(Node[i], sink)
                    ETx, _ = calculate_energy_consumption(Node[i], packetLength, direct_to_sink_dist)
                    Node[i].power -= ETx  # 只考虑发送能量，基站不考虑能量消耗

        total_delay = 0
        active_nodes = 0
        for i in range(n):
            if Node[i].flag != 0:
                delay = calculate_delay(Node[i], sink, packetLength, Node)
                total_delay += delay
                active_nodes += 1
        if active_nodes > 0:
            current_delay = total_delay / active_nodes
            temp_delays.append(current_delay)
            if (r + 1) % window_size == 0 and temp_delays:
                avg_delay = sum(temp_delays) / len(temp_delays)
                delay_improve_leach[delay_index] = avg_delay
                delay_index += 1
                temp_delays = []
            delay_improve_leach[r] = current_delay

        # 统计数据包传输
        round_packets_sent = 0
        for i in range(n):
            if Node[i].flag != 0:
                if Node[i].type == 'C':
                    round_packets_sent += len(Node[i].cluster_members) + 1
                elif Node[i].CH != -1:
                    round_packets_sent += 1

        pdr_improve_leach[r] = calculate_round_pdr(Node, sink, round_packets_sent)
        received_packets = round_packets_sent * (pdr_improve_leach[r] / 100)
        throughput_improve_leach[r] = calculate_throughput(received_packets, round_time)
        throughput_per_round.append(throughput_improve_leach[r])

        for i in range(n):
            if Node[i].flag != 0 and Node[i].power <= 0:
                Node[i].flag = 0
                Node[i].power = 0

        if is_display and (r + 1) % 10 == 0:
            pass # display_network(Node, sink, r)

        std = calculate_energy_std(Node)
        energy_std_data.append(std)

    if temp_delays:
        avg_delay = sum(temp_delays) / len(temp_delays)
        delay_improve_leach[delay_index] = avg_delay

    while len(false_wakeup_count_list) < rmax:
        false_wakeup_count_list.append(0)
    return alive_improve_leach, re_improve_leach, pdr_improve_leach, delay_improve_leach, energy_std_data, throughput_improve_leach, false_wakeup_count_list, false_wakeup_total, false_wakeup_energy

def display_network(Node, sink, r):
    """显示网络状态"""
    plt.cla()
    plt.grid(linestyle="dotted")
    plt.scatter(sink.x, sink.y, marker='*', s=200, color='red')

    # 绘制层级边界（可选）
    max_layer = max(node.layer for node in Node if node.flag != 0)
    layer_colors = plt.cm.rainbow(np.linspace(0, 1, max_layer + 1))

    # 绘制节点和连接
    for i in range(n):
        if Node[i].flag != 0:
            # 根据层级为节点设置不同颜色
            node_color = layer_colors[Node[i].layer]

            if Node[i].type == 'C':
                # 簇头节点
                plt.scatter(Node[i].xd, Node[i].yd, marker='^', color=node_color, s=100, edgecolor='black')

                # 显示簇成员到簇头的连接
                for member in Node[i].cluster_members:
                    plt.plot([Node[i].xd, Node[member].xd],
                             [Node[i].yd, Node[member].yd],
                             'k-', linewidth=0.5, alpha=0.3)

                # 找到这个簇头的网关连接（如果存在）
                found_gateway = False
                for gateway in [Node[j] for j in range(n) if Node[j].flag != 0 and Node[j].is_gateway]:
                    if gateway.layer < Node[i].layer:
                        # 检查是否是通过这个网关转发
                        for j in Node[i].N:
                            # 这里简化了判断逻辑，实际应使用与run_improve_leach中相同的网关选择逻辑
                            if j == gateway.id:
                                plt.plot([Node[i].xd, gateway.xd],
                                         [Node[i].yd, gateway.yd],
                                         'g--', linewidth=1.5)
                                found_gateway = True
                                break

                # 如果没有找到网关，直接连接到基站
                if not found_gateway:
                    plt.plot([Node[i].xd, sink.x],
                             [Node[i].yd, sink.y],
                             'r--', linewidth=1)

            elif Node[i].is_gateway:
                # 网关节点
                plt.scatter(Node[i].xd, Node[i].yd, marker='s', color=node_color, s=80, edgecolor='black')

                # 显示网关到基站或下一层网关的连接
                if Node[i].parent:
                    parent_id = Node[i].parent
                    if parent_id == -1:  # 直接连接到基站
                        plt.plot([Node[i].xd, sink.x],
                                 [Node[i].yd, sink.y],
                                 'b-', linewidth=1.5)
                    else:  # 连接到其他网关
                        plt.plot([Node[i].xd, Node[parent_id].xd],
                                 [Node[i].yd, Node[parent_id].yd],
                                 'b-', linewidth=1.5)
            else:
                # 普通节点
                plt.scatter(Node[i].xd, Node[i].yd, marker='o', color=node_color, s=30, alpha=0.7)

    # 在图例中添加层级信息
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='基站'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='簇头节点'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='网关节点'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='普通节点')
    ]

    # 添加连接类型的图例
    legend_elements.extend([
        plt.Line2D([0], [0], color='k', linewidth=0.5, alpha=0.3, label='簇内通信'),
        plt.Line2D([0], [0], color='g', linestyle='--', linewidth=1.5, label='簇头到网关'),
        plt.Line2D([0], [0], color='b', linewidth=1.5, label='网关间转发'),
        plt.Line2D([0], [0], color='r', linestyle='--', linewidth=1, label='直接到基站')
    ])

    # 添加层级图例
    for i in range(max_layer + 1):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=layer_colors[i], markersize=8,
                       label=f'第{i}层')
        )

    plt.legend(handles=legend_elements, prop={'family': 'SimHei'}, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xlabel('x坐标', fontproperties='SimHei', size=12)
    plt.ylabel('y坐标', fontproperties='SimHei', size=12)
    plt.title(f'改进LEACH协议：第{r + 1}轮', fontproperties='SimHei', size=15)
    plt.ion()
    plt.draw()
    plt.pause(0.15)
    plt.show(block=False)

    # 添加保存图片的代码
    plt.savefig(f'network_round_{r}.png')  # 保存每一轮的网络状态
    plt.close()  # 关闭图形，避免内存占用


def run_teen():
    """运行TEEN协议"""
    Node , _ = initialize_network()
    alive_teen = np.zeros(rmax)
    re_teen = np.zeros(rmax)
    pdr_teen = np.zeros(rmax)
    delay_teen = np.full(rmax, np.nan)
    throughput_teen = np.zeros(rmax)

    temp_delays = []
    window_size = 200
    delay_index = 0
    energy_std_data = []
    throughput_per_round = []
    round_time = 1  # 假设每轮持续1秒
    wakeup_reason = None

    HARD_THRESHOLD = 0.7
    SOFT_THRESHOLD = 0.3

    false_wakeup_count_list = []  # 每轮误唤醒节点数
    false_wakeup_total = 0  # 累计误唤醒次数
    false_wakeup_energy = 0  # 累计误唤醒能耗
    for r in range(rmax):
        # 统计存活节点和总能量
        for i in range(n):
            if Node[i].flag != 0:
                re_teen[r] += Node[i].power
                alive_teen[r] += 1
        if alive_teen[r] == 0:
            break

        # 重置节点状态
        for i in range(n):
            Node[i].type = 'N'
            Node[i].CH = 0
            Node[i].last_report_value = None if r == 0 else Node[i].last_report_value
            Node[i].last_report_time = 0 if r == 0 else Node[i].last_report_time
            Node[i].state = 'SLEEP'
            Node[i].has_uploaded = False

        # 簇头选举阶段（复用LEACH逻辑）
        for i in range(n):
            if Node[i].flag != 0:
                if random.random() <= LEACH_P * (1 - LEACH_P * (r % round(1 / LEACH_P))):
                    Node[i].type = 'C'
                    Node[i].CH = -1
                    # 广播簇头状态消息的能耗
                    broadcast_dist = Rmax
                    if broadcast_dist > d0:
                        broadcast_energy = (Eelec * ctrPacketLength +
                                            Emp * ctrPacketLength * (broadcast_dist ** 4))
                    else:
                        broadcast_energy = (Eelec * ctrPacketLength +
                                            Efs * ctrPacketLength * (broadcast_dist ** 2))
                    Node[i].power -= broadcast_energy

        # 簇成员加入阶段
        for i in range(n):
            if Node[i].flag != 0 and Node[i].type == 'N':
                min_dist = float('inf')
                selected_CH = -1
                for j in range(n):
                    if Node[j].type == 'C':
                        dist = calculate_distance(Node[i], Node[j])
                        if dist < min_dist:
                            min_dist = dist
                            selected_CH = j
                if selected_CH != -1:
                    Node[i].CH = selected_CH
                    # 发送加入请求的能耗
                    if min_dist > d0:
                        join_energy = (Eelec * ctrPacketLength +
                                       Emp * ctrPacketLength * (min_dist ** 4))
                    else:
                        join_energy = (Eelec * ctrPacketLength +
                                       Efs * ctrPacketLength * (min_dist ** 2))
                    Node[i].power -= join_energy
                    Node[selected_CH].power -= Eelec * ctrPacketLength

        # TDMA调度阶段（可省略或复用LEACH）
        for i in range(n):
            if Node[i].type == 'C':
                # 广播TDMA时隙表的能耗
                broadcast_dist = Rmax
                if broadcast_dist > d0:
                    schedule_energy = (Eelec * ctrPacketLength +
                                       Emp * ctrPacketLength * (broadcast_dist ** 4))
                else:
                    schedule_energy = (Eelec * ctrPacketLength +
                                       Efs * ctrPacketLength * (broadcast_dist ** 2))
                Node[i].power -= schedule_energy
                for j in range(n):
                    if Node[j].CH == i:
                        Node[j].power -= Eelec * ctrPacketLength

        # TEEN核心：节点感知与上报（周期性+事件驱动）
        false_wakeup_count = 0
        for i in range(n):
            if Node[i].flag != 0:
                value = simulate_sensor_data(Node[i], r)
                # 事件驱动：首次达到硬阈值或变化量超过软阈值
                if Node[i].last_report_value is None and value >= HARD_THRESHOLD:
                    Node[i].state = 'ACTIVE'
                    Node[i].ready_to_upload = True
                    Node[i].last_report_value = value
                    Node[i].last_report_time = r
                    Node[i].wakeup_reason = 'HARD_THRESHOLD'
                elif Node[i].last_report_value is not None and abs(value - Node[i].last_report_value) >= SOFT_THRESHOLD:
                    Node[i].state = 'ACTIVE'
                    Node[i].ready_to_upload = True
                    Node[i].last_report_value = value
                    Node[i].last_report_time = r
                    Node[i].wakeup_reason = 'SOFT_THRESHOLD'
                else:
                    Node[i].state = 'SLEEP'
                    Node[i].ready_to_upload = False
                    Node[i].wakeup_reason = None
                # 误唤醒判定
                if Node[i].state == 'ACTIVE' and Node[i].wakeup_reason == 'SOFT_THRESHOLD' and value < HARD_THRESHOLD:
                    false_wakeup_count += 1
                    false_wakeup_total += 1
                    false_wakeup_energy += WAKEUP_POWER
        false_wakeup_count_list.append(false_wakeup_count)

        # 数据传输阶段
        for i in range(n):
            if Node[i].flag != 0 and Node[i].state == 'ACTIVE':
                if Node[i].type == 'C':
                    # 簇头直接向基站发送
                    dist_to_sink = calculate_distance(Node[i], sink)
                    if dist_to_sink > d0:
                        transmission_energy = (Eelec * packetLength +
                                               Emp * packetLength * (dist_to_sink ** 4))
                    else:
                        transmission_energy = (Eelec * packetLength +
                                               Efs * packetLength * (dist_to_sink ** 2))
                    Node[i].power -= transmission_energy
                elif Node[i].CH != -1:
                    # 普通节点向簇头发送
                    dist = calculate_distance(Node[i], Node[Node[i].CH])
                    if dist > d0:
                        transmission_energy = (Eelec * packetLength +
                                               Emp * packetLength * (dist ** 4))
                    else:
                        transmission_energy = (Eelec * packetLength +
                                               Efs * packetLength * (dist ** 2))
                    Node[i].power -= transmission_energy
                    Node[Node[i].CH].power -= (Eelec + ED) * packetLength

                Node[i].has_uploaded = True
                Node[i].state = 'SLEEP'

        # 计算该轮的平均延时
        total_delay = 0
        active_nodes = 0
        for i in range(n):
            if Node[i].flag != 0:
                delay = calculate_delay(Node[i], sink, packetLength, Node)
                total_delay += delay
                active_nodes += 1
        if active_nodes > 0:
            current_delay = total_delay / active_nodes
            temp_delays.append(current_delay)
            if (r + 1) % window_size == 0 and temp_delays:
                avg_delay = sum(temp_delays) / len(temp_delays)
                delay_teen[r] = avg_delay  # 只在窗口末尾赋值
                temp_delays = []
            delay_teen[r] = current_delay

        # 统计数据包传输
        round_packets_sent = 0
        for i in range(n):
            if Node[i].flag != 0 and Node[i].state == 'ACTIVE':
                if Node[i].type == 'C':
                    round_packets_sent += 1
                elif Node[i].CH != -1:
                    round_packets_sent += 1

        # 计算PDR（复用LEACH的PDR计算方式）
        pdr_teen[r] = calculate_leach_pdr(Node, sink, round_packets_sent)
        received_packets = round_packets_sent * pdr_teen[r]
        throughput_teen[r] = calculate_throughput(received_packets, round_time)
        throughput_per_round.append(throughput_teen[r])

        # 检查节点死亡
        for i in range(n):
            if Node[i].flag != 0 and Node[i].power <= 0:
                Node[i].flag = 0
                Node[i].power = 0

        # 在每轮结束时记录标准差
        std = calculate_energy_std(Node)
        energy_std_data.append(std)

    if temp_delays:
        avg_delay = sum(temp_delays) / len(temp_delays)
        delay_teen[r] = avg_delay

    return alive_teen, re_teen, pdr_teen, delay_teen, energy_std_data, throughput_teen, false_wakeup_count_list, false_wakeup_total, false_wakeup_energy


# 节点复位 - 为LEACH协议准备
def run_leach():
    """运行LEACH协议"""
    Node , _ = initialize_network()
    alive_leach = np.zeros(rmax)
    re_leach = np.zeros(rmax)
    pdr_leach = np.zeros(rmax)
    delay_leach = np.zeros(rmax)
    throughput_leach = np.zeros(rmax)

    temp_delays = []
    window_size = 200
    delay_index = 0
    # 在函数开始处添加标准差记录列表
    energy_std_data = []
    throughput_per_round = []
    round_time = 1  # 假设每轮持续1秒

    for r in range(rmax):
        # 统计存活节点和总能量
        for i in range(n):
            if Node[i].flag != 0:
                re_leach[r] += Node[i].power
                alive_leach[r] += 1

        if alive_leach[r] == 0:
            # 如果所有节点死亡，则从当前轮次开始，PDR记录为0，并跳过本轮后续计算
            pdr_leach[r:] = 0
            throughput_leach[r:] = 0
            delay_leach[r:] = 0
            # 补充剩余轮次的能量标准差为nan
            for _ in range(r+1, rmax):
                energy_std_data.append(np.nan)
            break

        # 重置节点状态
        for i in range(n):
            Node[i].type = 'N'
            Node[i].CH = 0

        # 簇头选举阶段
        for i in range(n):
            if Node[i].flag != 0:
                # LEACH的簇头选举概率计算 /
                if random.random() <= LEACH_P * (1 - LEACH_P * (r % round(1 / LEACH_P))):
                    Node[i].type = 'C'
                    Node[i].CH = -1

                    # 广播簇头状态消息的能耗
                    broadcast_dist = Rmax
                    if broadcast_dist > d0:
                        broadcast_energy = (Eelec * ctrPacketLength +
                                            Emp * ctrPacketLength * (broadcast_dist ** 4))
                    else:
                        broadcast_energy = (Eelec * ctrPacketLength +
                                            Efs * ctrPacketLength * (broadcast_dist ** 2))
                    Node[i].power -= broadcast_energy

        # 簇成员加入阶段
        for i in range(n):
            if Node[i].flag != 0 and Node[i].type == 'N':
                min_dist = float('inf')
                selected_CH = -1

                # 选择最近的簇头
                for j in range(n):
                    if Node[j].type == 'C':
                        dist = calculate_distance(Node[i], Node[j])
                        if dist < min_dist:
                            min_dist = dist
                            selected_CH = j

                if selected_CH != -1:
                    Node[i].CH = selected_CH

                    # 发送加入请求的能耗
                    if min_dist > d0:
                        join_energy = (Eelec * ctrPacketLength +
                                       Emp * ctrPacketLength * (min_dist ** 4))
                    else:
                        join_energy = (Eelec * ctrPacketLength +
                                       Efs * ctrPacketLength * (min_dist ** 2))
                    Node[i].power -= join_energy

                    # 簇头接收加入请求的能耗
                    Node[selected_CH].power -= Eelec * ctrPacketLength

        
        for i in range(n):
            if Node[i].type == 'C':
                # 广播TDMA时隙表的能耗
                broadcast_dist = Rmax
                if broadcast_dist > d0:
                    schedule_energy = (Eelec * ctrPacketLength +
                                       Emp * ctrPacketLength * (broadcast_dist ** 4))
                else:
                    schedule_energy = (Eelec * ctrPacketLength +
                                       Efs * ctrPacketLength * (broadcast_dist ** 2))
                Node[i].power -= schedule_energy

                # 簇成员接收TDMA时隙表的能耗
                for j in range(n):
                    if Node[j].CH == i:
                        Node[j].power -= Eelec * ctrPacketLength

        # 数据传输阶段
        for i in range(n):
            if Node[i].flag != 0:
                if Node[i].type == 'C':
                    # 1. 簇头接收成员节点数据的能耗
                    member_count = sum(1 for node in range(n) if Node[node].CH == i)
                    receive_energy = member_count * (Eelec + ED) * packetLength
                    Node[i].power -= receive_energy

                    # 2. 簇头数据聚合的能耗
                    aggregation_energy = member_count * ED * packetLength
                    Node[i].power -= aggregation_energy

                    # 3. 簇头向基站发送聚合数据的能耗
                    dist_to_sink = calculate_distance(Node[i], sink)
                    if dist_to_sink > d0:
                        transmission_energy = (Eelec * packetLength +
                                               Emp * packetLength * (dist_to_sink ** 4))
                    else:
                        transmission_energy = (Eelec * packetLength +
                                               Efs * packetLength * (dist_to_sink ** 2))
                    Node[i].power -= transmission_energy

                else:  # 普通节点
                    if Node[i].CH >= 0:
                        # 向簇头发送数据的能耗
                        dist = calculate_distance(Node[i], Node[Node[i].CH])
                        if dist > d0:
                            transmission_energy = (Eelec * packetLength +
                                                   Emp * packetLength * (dist ** 4))
                        else:
                            transmission_energy = (Eelec * packetLength +
                                                   Efs * packetLength * (dist ** 2))
                        Node[i].power -= transmission_energy

        # 计算该轮的平均延时
        total_delay = 0
        active_nodes = 0
        for i in range(n):
            if Node[i].flag != 0:
                delay = calculate_delay(Node[i], sink, packetLength, Node)
                total_delay += delay
                active_nodes += 1

        if active_nodes > 0:
            current_delay = total_delay / active_nodes
            temp_delays.append(current_delay)

            if (r + 1) % window_size == 0 and temp_delays:
                avg_delay = sum(temp_delays) / len(temp_delays)
                delay_leach[delay_index] = avg_delay
                delay_index += 1
                temp_delays = []
            delay_leach[r] = current_delay

        # 计算PDR
        round_packets_sent = 0
        for i in range(n):
            if Node[i].flag != 0:
                if Node[i].type == 'C':
                    round_packets_sent += len(Node[i].cluster_members) + 1  # 簇头和簇成员的数据包
                elif Node[i].CH != -1:
                    round_packets_sent += 1  # 簇成员的数据包

        # 计算PDR
        pdr_leach[r] = calculate_leach_pdr(Node, sink, round_packets_sent)
        received_packets = round_packets_sent * (pdr_leach[r] / 100)
        throughput_leach[r] = calculate_throughput(received_packets, round_time)
        throughput_per_round.append(throughput_leach[r])

        # 检查节点死亡
        for i in range(n):
            if Node[i].flag != 0 and Node[i].power <= 0:
                Node[i].flag = 0
                Node[i].power = 0

        # 动态显示
        if is_display and (r + 1) % 10 == 0:
            pass # display_leach_network(Node, sink, r)

        # 在每轮结束时记录标准差
        std = calculate_energy_std(Node)
        energy_std_data.append(std)

    if temp_delays:
        avg_delay = sum(temp_delays) / len(temp_delays)
        delay_leach[delay_index] = avg_delay

    return alive_leach, re_leach, pdr_leach, delay_leach, energy_std_data, throughput_leach


def display_leach_network(Node, sink, r):
    """显示LEACH网络状态"""
    pass


def run_heed():
    """运行HEED协议的实验"""
    Node , _ = initialize_network()
    alive_heed = np.zeros(rmax)
    re_heed = np.zeros(rmax)
    pdr_heed = np.zeros(rmax)
    delay_heed = np.zeros(rmax)
    throughput_heed = np.zeros(rmax)

    temp_delays = []
    window_size = 200
    delay_index = 0
    # 在函数开始处添加标准差记录列表
    energy_std_data = []
    throughput_per_round = []
    round_time = 1  # 假设每轮持续1秒

    def calculate_amrp(node, neighbors, Node):
        """计算平均最小可达功率(AMRP)"""
        if not neighbors:
            return float('inf')
        total_min_power = 0
        for neighbor in neighbors:
            dist = calculate_distance(node, Node[neighbor])
            # 计算与邻居通信所需的最小功率
            if dist <= d0:
                min_power = Eelec * packetLength + Efs * packetLength * (dist ** 2)
            else:
                min_power = Eelec * packetLength + Emp * packetLength * (dist ** 4)
            total_min_power += min_power
        return total_min_power / len(neighbors)

    for r in range(rmax):
        # 统计存活节点和总能量
        for i in range(n):
            if Node[i].flag != 0:
                re_heed[r] += Node[i].power
                alive_heed[r] += 1

        if alive_heed[r] == 0:
            # 若节点全死，后续补nan直到rmax
            for _ in range(r+1, rmax):
                energy_std_data.append(np.nan)
            break

        # 重置节点状态
        for i in range(n):
            Node[i].type = 'N'
            Node[i].selected = 'N'
            Node[i].CH = 0
            Node[i].is_tentative = False
            Node[i].amrp = float('inf')

        tentative_CH = []
        final_CH = []

        # 初始化阶段
        for i in range(n):
            if Node[i].flag != 0:
                # 计算邻居节点
                neighbors = [j for j in range(n) if
                             Node[j].flag != 0 and i != j and
                             calculate_distance(Node[i], Node[j]) <= Rmax]

                # 计算AMRP值
                Node[i].amrp = calculate_amrp(Node[i], neighbors, Node)

                # 初始簇头概率
                CHprob = max(HEED_Cprob * Node[i].power / E0, HEED_Pmin)
                Node[i].temp_rand = CHprob

                if random.random() <= CHprob:
                    Node[i].is_tentative = True
                    tentative_CH.append(i)

                # 广播初始状态的能耗
                broadcast_dist = Rmax
                if broadcast_dist > d0:
                    broadcast_energy = (Eelec * ctrPacketLength +
                                        Emp * ctrPacketLength * (broadcast_dist ** 4))
                else:
                    broadcast_energy = (Eelec * ctrPacketLength +
                                        Efs * ctrPacketLength * (broadcast_dist ** 2))
                Node[i].power -= broadcast_energy

        # 迭代阶段
        for iteration in range(HEED_ITERATIONS):
            for i in tentative_CH[:]:
                # 获取邻居中的临时簇头
                neighbor_chs = [j for j in tentative_CH if j != i and
                                calculate_distance(Node[i], Node[j]) <= Rmax]

                if neighbor_chs:
                    # 比较AMRP值
                    if all(Node[i].amrp <= Node[j].amrp for j in neighbor_chs):
                        if Node[i].temp_rand >= 1:  # CHprob = 1
                            Node[i].type = 'C'
                            Node[i].selected = 'O'
                            Node[i].CH = -1
                            final_CH.append(i)
                            tentative_CH.remove(i)

                            # 广播最终簇头状态的能耗
                            broadcast_dist = Rmax
                            if broadcast_dist > d0:
                                broadcast_energy = (Eelec * ctrPacketLength +
                                                    Emp * ctrPacketLength * (broadcast_dist ** 4))
                            else:
                                broadcast_energy = (Eelec * ctrPacketLength +
                                                    Efs * ctrPacketLength * (broadcast_dist ** 2))
                            Node[i].power -= broadcast_energy
                else:
                    # 没有邻居簇头，增加簇头概率
                    Node[i].temp_rand = min(2 * Node[i].temp_rand, 1)

        # 最终簇头选择
        for i in tentative_CH:
            neighbor_chs = [j for j in tentative_CH if j != i and
                            calculate_distance(Node[i], Node[j]) <= Rmax]

            if not neighbor_chs or all(Node[i].amrp <= Node[j].amrp for j in neighbor_chs):
                Node[i].type = 'C'
                Node[i].selected = 'O'
                Node[i].CH = -1
                final_CH.append(i)

                # 广播最终簇头状态的能耗
                broadcast_dist = Rmax
                if broadcast_dist > d0:
                    broadcast_energy = (Eelec * ctrPacketLength +
                                        Emp * ctrPacketLength * (broadcast_dist ** 4))
                else:
                    broadcast_energy = (Eelec * ctrPacketLength +
                                        Efs * ctrPacketLength * (broadcast_dist ** 2))
                Node[i].power -= broadcast_energy

        # 簇成员加入
        for i in range(n):
            if Node[i].flag != 0 and Node[i].type == 'N':
                min_amrp = float('inf')
                selected_CH = -1

                for ch in final_CH:
                    if calculate_distance(Node[i], Node[ch]) <= Rmax:
                        if Node[ch].amrp < min_amrp:
                            min_amrp = Node[ch].amrp
                            selected_CH = ch

                if selected_CH != -1:
                    Node[i].CH = selected_CH

                    # 发送加入簇请求的能耗
                    dist = calculate_distance(Node[i], Node[selected_CH])
                    if dist > d0:
                        join_energy = (Eelec * ctrPacketLength +
                                       Emp * ctrPacketLength * (dist ** 4))
                    else:
                        join_energy = (Eelec * ctrPacketLength +
                                       Efs * ctrPacketLength * (dist ** 2))
                    Node[i].power -= join_energy

                    # 簇头接收加入请求的能耗
                    Node[selected_CH].power -= Eelec * ctrPacketLength

        # 数据传输阶段
        for i in range(n):
            if Node[i].flag != 0:
                if Node[i].type == 'C':
                    # 1. 簇头接收成员节点数据的能耗
                    member_count = sum(1 for node in range(n) if Node[node].CH == i)
                    receive_energy = member_count * (Eelec + ED) * packetLength
                    Node[i].power -= receive_energy

                    # 2. 簇头数据聚合的能耗
                    aggregation_energy = member_count * ED * packetLength
                    Node[i].power -= aggregation_energy

                    # 3. 簇头向基站发送聚合数据的能耗
                    dist_to_sink = calculate_distance(Node[i], sink)
                    if dist_to_sink > d0:
                        transmission_energy = (Eelec * packetLength +
                                               Emp * packetLength * (dist_to_sink ** 4))
                    else:
                        transmission_energy = (Eelec * packetLength +
                                               Efs * packetLength * (dist_to_sink ** 2))
                    Node[i].power -= transmission_energy

                    # 4. 簇头广播TDMA时隙表的能耗
                    broadcast_dist = Rmax
                    if broadcast_dist > d0:
                        broadcast_energy = (Eelec * ctrPacketLength +
                                            Emp * ctrPacketLength * (broadcast_dist ** 4))
                    else:
                        broadcast_energy = (Eelec * ctrPacketLength +
                                            Efs * ctrPacketLength * (broadcast_dist ** 2))
                    Node[i].power -= broadcast_energy

                else:  # 普通节点
                    if Node[i].CH >= 0:
                        # 1. 接收TDMA时隙表的能耗
                        Node[i].power -= Eelec * ctrPacketLength

                        # 2. 向簇头发送数据的能耗
                        dist = calculate_distance(Node[i], Node[Node[i].CH])
                        if dist > d0:
                            transmission_energy = (Eelec * packetLength +
                                                   Emp * packetLength * (dist ** 4))
                        else:
                            transmission_energy = (Eelec * packetLength +
                                                   Efs * packetLength * (dist ** 2))
                        Node[i].power -= transmission_energy

        # 计算该轮的平均延时
        total_delay = 0
        active_nodes = 0
        for i in range(n):
            if Node[i].flag != 0:
                delay = calculate_delay(Node[i], sink, packetLength, Node)
                total_delay += delay
                active_nodes += 1

        if active_nodes > 0:
            current_delay = total_delay / active_nodes
            temp_delays.append(current_delay)

            if (r + 1) % window_size == 0 and temp_delays:
                avg_delay = sum(temp_delays) / len(temp_delays)
                delay_heed[delay_index] = avg_delay
                delay_index += 1
                temp_delays = []
            delay_heed[r] = current_delay

        # 统计数据包数量
        round_packets_sent = 0
        for i in range(n):
            if Node[i].flag != 0:
                if Node[i].type == 'C':
                    round_packets_sent += len(Node[i].cluster_members) + 1  # 簇头和簇成员的数据包
                elif Node[i].CH != -1:
                    round_packets_sent += 1  # 簇成员的数据包

        # 计算PDR
        pdr_heed[r] = calculate_heed_pdr(Node, sink, round_packets_sent)
        received_packets = round_packets_sent * (pdr_heed[r] / 100)
        throughput_heed[r] = calculate_throughput(received_packets, round_time)
        throughput_per_round.append(throughput_heed[r])

        # 检查节点死亡
        for i in range(n):
            if Node[i].flag != 0 and Node[i].power <= 0:
                Node[i].flag = 0
                Node[i].power = 0

        # 动态显示
        if is_display and (r + 1) % 10 == 0:
            pass # display_heed_network(Node, sink, r)

        # 在每轮结束时记录标准差
        std = calculate_energy_std(Node)
        energy_std_data.append(std)

    if temp_delays:
        avg_delay = sum(temp_delays) / len(temp_delays)
        delay_heed[delay_index] = avg_delay

    return alive_heed, re_heed, pdr_heed, delay_heed, energy_std_data, throughput_heed


def display_heed_network(Node, sink, r):
    """显示HEED网络状态"""
    pass


def run_d2crp():
    """运行D2CRP协议"""
    # 初始化统计数组
    alive_d2crp = np.zeros(rmax)
    re_d2crp = np.zeros(rmax)
    pdr_d2crp = np.zeros(rmax)
    delay_d2crp = np.zeros(rmax)
    throughput_d2crp = np.zeros(rmax)

    temp_delays = []
    window_size = 200
    delay_index = 0
    # 在函数开始处添加标准差记录列表
    energy_std_data = []
    throughput_per_round = []
    round_time = 1  # 假设每轮持续1秒

    CLUSTER_COUNT = 16  # 固定簇数
    network_area = xm * ym
    fixed_distance = math.sqrt(network_area) / (8 * math.pi * CLUSTER_COUNT)

    # 初始化节点
    Node = [D2CRPNode() for _ in range(n)]
    for i in range(n):
        Node[i].id = i
        Node[i].xd = random.random() * xm
        Node[i].yd = random.random() * ym
        Node[i].power = E0
        Node[i].init_power = E0
        Node[i].flag = 1
        Node[i].type = 'N'

    def broadcast_node_info():
        """节点广播阶段"""
        for i in range(n):
            if Node[i].flag == 0:
                continue

            # 计算与基站距离
            dist_to_sink = calculate_distance(Node[i], sink)

            # 广播能耗
            ETx, ERx = calculate_energy_consumption(
                Node[i], ctrPacketLength, fixed_distance
            )
            Node[i].power -= ETx  # 发送能量

            # 收集1跳邻居
            Node[i].one_hop.clear()
            for j in range(n):
                if i != j and Node[j].flag != 0:
                    dist = calculate_distance(Node[i], Node[j])
                    if dist <= fixed_distance:
                        Node[i].one_hop.add(j)
                        # 接收能耗
                        Node[j].power -= ERx  # 接收能量

            # 收集2跳邻居
            Node[i].two_hop.clear()
            for j in Node[i].one_hop:
                for k in Node[j].one_hop:
                    if k != i and k not in Node[i].one_hop:
                        Node[i].two_hop.add(k)

    def calculate_factors():
        """计算距离因子和能量因子"""
        for i in range(n):
            if Node[i].flag == 0:
                continue

            # 获取2跳域内所有节点
            two_hop_domain = Node[i].one_hop.union(Node[i].two_hop)
            if not two_hop_domain:
                Node[i].distance_factor = 1
                Node[i].energy_factor = 1
                continue

            # 计算距离因子
            dist_to_sink = calculate_distance(Node[i], sink)
            max_dist = max(calculate_distance(Node[j], sink)
                           for j in two_hop_domain)
            if max_dist > 0:
                Node[i].distance_factor = dist_to_sink / max_dist
            else:
                Node[i].distance_factor = 1

            # 计算能量因子
            max_energy = max(Node[j].power for j in two_hop_domain)
            # 防止除零
            if max_energy > 0:
                Node[i].energy_factor = Node[i].power / max_energy
            else:
                Node[i].energy_factor = 1

    def select_cluster_heads(round_num):
        """选举簇头"""
        cluster_heads = []

        for i in range(n):
            if Node[i].flag == 0 or Node[i].power < Emin:
                continue

            if Node[i].type != 'C':
                temp_rand = random.random()
                threshold = (D2CRP_P / (1 - D2CRP_P * (round_num % round(1 / D2CRP_P)))) * \
                            (Node[i].energy_factor / Node[i].distance_factor)

                if temp_rand <= threshold:
                    Node[i].type = 'C'
                    Node[i].CH = None
                    cluster_heads.append(i)

                    # 广播簇头状态
                    ETx, ERx = calculate_energy_consumption(
                        Node[i], ctrPacketLength, fixed_distance
                    )
                    Node[i].power -= ETx  # 发送能量
                    
                    # 邻居节点接收广播
                    for j in Node[i].one_hop:
                        Node[j].power -= ERx  # 接收能量

        return cluster_heads

    def join_clusters(cluster_heads):
        """节点加入簇过程"""
        n_per_cluster = n // CLUSTER_COUNT
        one_hop_count = int(0.25 * (n_per_cluster - 1))

        # 1. 一跳节点加入
        for i in range(n):
            if Node[i].flag == 0 or Node[i].type == 'C':
                continue

            min_dist = float('inf')
            selected_ch = None

            for ch in cluster_heads:
                if ch in Node[i].one_hop:
                    dist = calculate_distance(Node[i], Node[ch])
                    if dist < min_dist:
                        min_dist = dist
                        selected_ch = ch

            if selected_ch is not None and len(Node[selected_ch].cluster_members) < one_hop_count:
                Node[i].type = '1-hop'
                Node[i].CH = selected_ch
                Node[selected_ch].cluster_members.append(i)

        # 2. 二跳节点加入
        for i in range(n):
            if Node[i].flag == 0 or Node[i].type != 'N':
                continue

            min_dist = float('inf')
            selected_relay = None

            for j in Node[i].one_hop:
                if Node[j].type == '1-hop':
                    dist = calculate_distance(Node[i], Node[j])
                    if dist < min_dist:
                        min_dist = dist
                        selected_relay = j

            if selected_relay is not None:
                Node[i].type = '2-hop'
                Node[i].CH = Node[selected_relay].CH
                Node[i].relay_node = selected_relay
                Node[selected_relay].cluster_members.append(i)

    def setup_routing(cluster_heads):
        """建立簇间路由"""
        sqrt_m = int(math.sqrt(CLUSTER_COUNT))

        # 将CH按到基站距离排序
        sorted_ch = sorted(cluster_heads,
                           key=lambda x: calculate_distance(Node[x], sink))

        # 最近的√m个CH直接连接基站
        for i in range(min(sqrt_m, len(sorted_ch))):
            Node[sorted_ch[i]].next_hop = None  # None表示直接连接基站

        # 其余CH寻找下一跳
        for i in range(sqrt_m, len(sorted_ch)):
            ch = sorted_ch[i]
            min_dist = calculate_distance(Node[ch], sink)
            next_hop = None

            # 在2跳范围内寻找更近的CH
            for j in range(i):
                other_ch = sorted_ch[j]
                if other_ch in (Node[ch].one_hop.union(Node[ch].two_hop)):
                    dist_to_sink = calculate_distance(Node[other_ch], sink)
                    if dist_to_sink < min_dist:
                        min_dist = dist_to_sink
                        next_hop = other_ch

            Node[ch].next_hop = next_hop

    def data_transmission(cluster_heads):
        round_packets_sent = 0
        round_packets_received = 0
        """数据传输阶段"""
        m = CLUSTER_COUNT
        sqrt_m = int(math.sqrt(m))
        n_per_cluster = n // m

        # 1. 簇内数据传输
        for ch in cluster_heads:
            if Node[ch].flag == 0:
                continue

            # 统计簇内节点
            two_hop_nodes = [member for member in Node[ch].cluster_members
                             if Node[member].type == '2-hop']
            one_hop_nodes = [member for member in Node[ch].cluster_members
                             if Node[member].type == '1-hop']

            # 2跳节点传输到1跳节点
            two_hop_count = int(0.75 * (n_per_cluster - 1))
            for node in two_hop_nodes[:two_hop_count]:
                if Node[node].relay_node:
                    round_packets_sent += 1
                    # 2跳节点到1跳节点的能耗
                    ETx, ERx = calculate_energy_consumption(Node[node], packetLength, fixed_distance)
                    Node[node].power -= ETx  # 2跳节点发送能耗
                    Node[Node[node].relay_node].power -= ERx  # 1跳节点接收能耗
                    round_packets_received += 1

            # 1跳节点传输到CH
            one_hop_count = int(0.25 * (n_per_cluster - 1))
            for node in one_hop_nodes[:one_hop_count]:
                round_packets_sent += 1
                # 1跳节点发送能耗

                tx_energy = packetLength * (Eelec + Efs * fixed_distance * fixed_distance)
                Node[node].power -= tx_energy

                # CH接收能耗
                rx_energy = packetLength * Eelec
                Node[ch].power -= rx_energy
                round_packets_received += 1

        # 2. 簇间数据传输
        sorted_ch = sorted(cluster_heads,
                           key=lambda x: calculate_distance(Node[x], sink))
        L = 50  # 固定距离L

        # 最近的√m个CH直接传输到基站
        for ch in sorted_ch[:sqrt_m]:
            round_packets_sent += 1
            tx_energy = sqrt_m * packetLength * (Eelec + Efs * L * L)
            Node[ch].power -= tx_energy
            round_packets_received += 1

            # 其余CH的多跳传输
        ch_hop_distance = (4 + math.sqrt(2)) * math.sqrt(network_area) / (6 * sqrt_m)
        for ch in sorted_ch[sqrt_m:]:
            round_packets_sent += 1
            if Node[ch].next_hop is not None:
                # CH发送能耗
                tx_energy = (m - sqrt_m) * packetLength * \
                            (Eelec + Efs * ch_hop_distance * ch_hop_distance)
                Node[ch].power -= tx_energy

                # 接收CH能耗
                rx_energy = (m - sqrt_m) * packetLength * Eelec
                Node[Node[ch].next_hop].power -= rx_energy
                round_packets_received += 1
        return round_packets_sent, round_packets_received

    def calculate_pdr(cluster_heads):
        """计算PDR"""
        round_packets_sent = 0
        round_packets_received = 0

        # 簇内PDR计算
        for ch in cluster_heads:
            if Node[ch].flag == 0:
                continue

            # 二跳节点的PDR
            two_hop_nodes = [member for member in Node[ch].cluster_members
                             if Node[member].type == '2-hop']
            for node in two_hop_nodes:
                if Node[node].flag != 0:  # 只统计活跃节点
                    round_packets_sent += 1
                    if (Node[node].power > Emin and
                            Node[node].relay_node and
                            Node[Node[node].relay_node].flag != 0):
                        # 放宽链路质量计算
                        dist = calculate_distance(Node[node], Node[Node[node].relay_node])
                        link_quality = max(0.3, 1 - (dist / (3 * fixed_distance))) 
                        if random.random() < link_quality:
                            round_packets_received += 1

            # 一跳节点的PDR
            one_hop_nodes = [member for member in Node[ch].cluster_members
                             if Node[member].type == '1-hop']
            for node in one_hop_nodes:
                if Node[node].flag != 0:  # 只统计活跃节点
                    round_packets_sent += 1
                    if (Node[node].power > Emin and
                            Node[ch].flag != 0):
                        # 放宽链路质量计算
                        dist = calculate_distance(Node[node], Node[ch])
                        link_quality = max(0.4, 1 - (dist / (2 * fixed_distance)))  
                        if random.random() < link_quality:
                            round_packets_received += 1

        # 簇间PDR计算
        for ch in cluster_heads:
            if Node[ch].flag != 0:  # 只统计活跃簇头
                round_packets_sent += 1
                if Node[ch].next_hop is None:
                    # 直接发送到基站
                    dist = calculate_distance(Node[ch], sink)
                    link_quality = max(0.5, 1 - (dist / (4 * fixed_distance)))  
                    if Node[ch].power > Emin:
                        if random.random() < link_quality:
                            round_packets_received += 1
                else:
                    # 通过其他CH转发
                    if (Node[ch].power > Emin and
                            Node[Node[ch].next_hop].flag != 0):
                        dist = calculate_distance(Node[ch], Node[Node[ch].next_hop])
                        link_quality = max(0.4, 1 - (dist / (3 * fixed_distance)))  
                        if random.random() < link_quality:
                            round_packets_received += 1

        # 防止除零错误
        if round_packets_sent == 0:
            return 0

        return round_packets_received / round_packets_sent

    # 主循环
    for r in range(rmax):
        # 统计存活节点和总能量
        alive_count = 0
        total_energy = 0
        for i in range(n):
            if Node[i].flag != 0:
                alive_count += 1
                total_energy += Node[i].power

        alive_d2crp[r] = alive_count
        re_d2crp[r] = total_energy

        if alive_count == 0:
            # 若节点全死，后续补nan直到rmax
            for _ in range(r+1, rmax):
                energy_std_data.append(np.nan)
            break

        # 重置节点状态
        for i in range(n):
            if Node[i].flag != 0:
                Node[i].type = 'N'
                Node[i].CH = None
                Node[i].cluster_members = []
                Node[i].next_hop = None
                Node[i].relay_node = None

        # 运行协议各阶段
        broadcast_node_info()
        calculate_factors()
        cluster_heads = select_cluster_heads(r)
        join_clusters(cluster_heads)
        setup_routing(cluster_heads)

        round_packets_sent, round_packets_received = data_transmission(cluster_heads)
        if round_packets_sent > 0:
            pdr_d2crp[r] = round_packets_received / round_packets_sent
        else:
            pdr_d2crp[r] = 0
        pdr_d2crp[r] = calculate_pdr(cluster_heads)
        throughput_d2crp[r] = calculate_throughput(round_packets_received, round_time)
        throughput_per_round.append(throughput_d2crp[r])

        # 检查节点死亡
        for i in range(n):
            if Node[i].flag != 0 and Node[i].power <= 0:
                Node[i].flag = 0
                Node[i].power = 0

        # 计算性能指标

        if r % 200 == 0:
            # 时延计算
            delays = [calculate_delay(Node[i], sink, packetLength, Node)
                      for i in range(n) if Node[i].flag != 0]
            if delays:
                delay_d2crp[r] = sum(delays) / len(delays)

        # 动态显示
        if is_display and (r + 1) % 10 == 0:
            pass # display_d2crp_network(Node, sink, r)

        # 在每轮结束时记录标准差
        std = calculate_energy_std(Node)
        energy_std_data.append(std)

    return alive_d2crp, re_d2crp, pdr_d2crp, delay_d2crp, energy_std_data, throughput_d2crp

def display_d2crp_network(Node, sink, r):
    """显示D2CRP网络状态"""
    pass


def run_apteen():
    """运行APTEEN协议（周期性+事件驱动）"""
    Node ,_ = initialize_network()
    alive_apteen = np.zeros(rmax)
    re_apteen = np.zeros(rmax)
    pdr_apteen = np.zeros(rmax)
    delay_apteen = np.full(rmax, np.nan)
    throughput_apteen = np.zeros(rmax)

    temp_delays = []
    window_size = 200
    delay_index = 0
    energy_std_data = []
    throughput_per_round = []
    round_time = 1  # 假设每轮持续1秒
    wakeup_reason = None
    HARD_THRESHOLD = 0.7
    SOFT_THRESHOLD = 0.3
    CTI = 10  # 周期性上报间隔（轮）

    false_wakeup_count_list = []  # 每轮误唤醒节点数
    false_wakeup_total = 0  # 累计误唤醒次数
    false_wakeup_energy = 0  # 累计误唤醒能耗
    for r in range(rmax):
        # 统计存活节点和总能量
        for i in range(n):
            if Node[i].flag != 0:
                re_apteen[r] += Node[i].power
                alive_apteen[r] += 1
        if alive_apteen[r] == 0:
            break

        # 重置节点状态
        for i in range(n):
            Node[i].type = 'N'
            Node[i].CH = 0
            Node[i].last_report_value = None if r == 0 else Node[i].last_report_value
            Node[i].last_report_time = 0 if r == 0 else Node[i].last_report_time
            Node[i].state = 'SLEEP'
            Node[i].has_uploaded = False

        # 簇头选举阶段（复用LEACH逻辑）
        for i in range(n):
            if Node[i].flag != 0:
                if random.random() <= LEACH_P * (1 - LEACH_P * (r % round(1 / LEACH_P))):
                    Node[i].type = 'C'
                    Node[i].CH = -1
                    # 广播簇头状态消息的能耗
                    broadcast_dist = Rmax
                    if broadcast_dist > d0:
                        broadcast_energy = (Eelec * ctrPacketLength +
                                            Emp * ctrPacketLength * (broadcast_dist ** 4))
                    else:
                        broadcast_energy = (Eelec * ctrPacketLength +
                                            Efs * ctrPacketLength * (broadcast_dist ** 2))
                    Node[i].power -= broadcast_energy

        # TDMA调度阶段（可省略或复用LEACH）
        for i in range(n):
            if Node[i].type == 'C':
                # 广播TDMA时隙表的能耗
                broadcast_dist = Rmax
                if broadcast_dist > d0:
                    schedule_energy = (Eelec * ctrPacketLength +
                                       Emp * ctrPacketLength * (broadcast_dist ** 4))
                else:
                    schedule_energy = (Eelec * ctrPacketLength +
                                       Efs * ctrPacketLength * (broadcast_dist ** 2))
                Node[i].power -= schedule_energy
                for j in range(n):
                    if Node[j].CH == i:
                        Node[j].power -= Eelec * ctrPacketLength

        # APTEEN核心：节点感知与上报（周期性+事件驱动）
        false_wakeup_count = 0
        for i in range(n):
            if Node[i].flag != 0:
                value = simulate_sensor_data(Node[i], r)
                # 事件驱动：首次达到硬阈值或变化量超过软阈值
                if Node[i].last_report_value is None and value >= HARD_THRESHOLD:
                    Node[i].state = 'ACTIVE'
                    Node[i].last_report_value = value
                    Node[i].last_report_time = r
                    Node[i].wakeup_reason = 'HARD_THRESHOLD'
                elif Node[i].last_report_value is not None and abs(value - Node[i].last_report_value) >= SOFT_THRESHOLD:
                    Node[i].state = 'ACTIVE'
                    Node[i].last_report_value = value
                    Node[i].last_report_time = r
                    Node[i].wakeup_reason = 'SOFT_THRESHOLD'
                # 周期性上报
                elif Node[i].last_report_time is not None and (r - Node[i].last_report_time) >= CTI:
                    Node[i].state = 'ACTIVE'
                    Node[i].last_report_value = value
                    Node[i].last_report_time = r
                    Node[i].wakeup_reason = 'PERIODIC'
                else:
                    Node[i].state = 'SLEEP'
                    Node[i].wakeup_reason = None
                # 误唤醒判定
                if Node[i].state == 'ACTIVE' and Node[i].wakeup_reason in ['SOFT_THRESHOLD',
                                                                           'PERIODIC'] and value < HARD_THRESHOLD:
                    false_wakeup_count += 1
                    false_wakeup_total += 1
                    false_wakeup_energy += WAKEUP_POWER
        false_wakeup_count_list.append(false_wakeup_count)

        # 数据传输阶段
        for i in range(n):
            if Node[i].flag != 0 and Node[i].state == 'ACTIVE':
                if Node[i].type == 'C':
                    # 簇头直接向基站发送
                    dist_to_sink = calculate_distance(Node[i], sink)
                    if dist_to_sink > d0:
                        transmission_energy = (Eelec * packetLength +
                                               Emp * packetLength * (dist_to_sink ** 4))
                    else:
                        transmission_energy = (Eelec * packetLength +
                                               Efs * packetLength * (dist_to_sink ** 2))
                    Node[i].power -= transmission_energy
                elif Node[i].CH != -1:
                    # 普通节点向簇头发送
                    dist = calculate_distance(Node[i], Node[Node[i].CH])
                    if dist > d0:
                        transmission_energy = (Eelec * packetLength +
                                               Emp * packetLength * (dist ** 4))
                    else:
                        transmission_energy = (Eelec * packetLength +
                                               Efs * packetLength * (dist ** 2))
                    Node[i].power -= transmission_energy
                    Node[Node[i].CH].power -= (Eelec + ED) * packetLength
                Node[i].state = 'SLEEP'

        # 计算该轮的平均延时
        total_delay = 0
        active_nodes = 0
        for i in range(n):
            if Node[i].flag != 0:
                delay = calculate_delay(Node[i], sink, packetLength, Node)
                total_delay += delay
                active_nodes += 1
        if active_nodes > 0:
            current_delay = total_delay / active_nodes
            temp_delays.append(current_delay)
            if (r + 1) % window_size == 0 and temp_delays:
                avg_delay = sum(temp_delays) / len(temp_delays)
                delay_apteen[delay_index] = avg_delay  # 只在窗口末尾赋值
                temp_delays = []
            delay_apteen[r] = current_delay

        # 统计数据包传输
        round_packets_sent = 0
        for i in range(n):
            if Node[i].flag != 0 and Node[i].state == 'ACTIVE':
                if Node[i].type == 'C':
                    round_packets_sent += 1
                elif Node[i].CH != -1:
                    round_packets_sent += 1

        # 计算PDR（复用LEACH的PDR计算方式）
        pdr_apteen[r] = calculate_leach_pdr(Node, sink, round_packets_sent)
        received_packets = round_packets_sent * pdr_apteen[r]
        throughput_apteen[r] = calculate_throughput(received_packets, round_time)
        throughput_per_round.append(throughput_apteen[r])

        # 检查节点死亡
        for i in range(n):
            if Node[i].flag != 0 and Node[i].power <= 0:
                Node[i].flag = 0
                Node[i].power = 0

        # 在每轮结束时记录标准差
        std = calculate_energy_std(Node)
        energy_std_data.append(std)

    if temp_delays:
        avg_delay = sum(temp_delays) / len(temp_delays)
        delay_apteen[delay_index] = avg_delay

    return alive_apteen, re_apteen, pdr_apteen, delay_apteen, energy_std_data, throughput_apteen, false_wakeup_count_list, false_wakeup_total, false_wakeup_energy


def run_weight_experiments():
    """运行不同权重配置的实验"""
    configs = ['original', 'energy', 'buffer', 'connection', 'balanced', 'distance']
    # 修正configs，使其与select_gateway_nodes中的key保持一致
    configs = ['original', 'distance', 'energy', 'buffer', 'hybrid', 'fuzzy']
    results = {}

    for config_name in configs:
        print(f"正在测试权重配置: {config_name}")
        # 只运行一次实验
        alive, re, pdr, delay, energy_std, throughput, false_wakeup_count_list, false_wakeup_total, false_wakeup_energy = run_improve_leach(
            config={'weight_config': config_name}) # 传递权重配置

        # 计算各项指标（排除0值）
        results[config_name] = {
            'throughput': np.mean([x for x in throughput if x > 0]),
            'delay': np.mean([x for x in delay if x > 0]),
            'pdr': np.mean([x for x in pdr if x > 0]),
            'energy': np.mean([x for x in re if x > 0]),
            'alive': np.mean([x for x in alive if x > 0]),
            'false_wakeup_count': np.mean([x for x in false_wakeup_count_list if x > 0]),
            'false_wakeup_total': false_wakeup_total,
            'false_wakeup_energy': false_wakeup_energy,
            # 添加时间序列数据
            'pdr_series': pdr,
            'energy_series': re,
            'alive_series': alive
        }

    return results

def plot_weight_experiment_line_charts(results):
    """
    绘制不同权重配置下的PDR、剩余能量和存活节点数的折线图。
    """
    plt.figure(figsize=(18, 6))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 确保中文显示
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    # 定义颜色列表，确保有足够的颜色与配置匹配
    colors = ['#D4EEFF', '#FFE5D6', '#FFF2CD', '#FF9B9B', '#44CA79', '#FEC7E4']

    # 映射权重配置的显示名称和对应的颜色
    # 按照 select_gateway_nodes 中的key顺序，并提供表格中的显示名称
    configs_to_plot_with_names = [
        ('original', '(0.4,0.3,0.3)'),
        ('distance', '(0.3,0.4,0.3)'),
        ('energy', '(0.3,0.3,0.4)'),
        ('buffer', '(0.6,0.2,0.2)'),
        ('hybrid', '(0.5,0.3,0.2)'), 
        ('fuzzy', '(0.4,0.4,0.2)')    
    ]

    # PDR 折线图
    plt.subplot(1, 3, 1)
    for i, (config_key, config_display_name) in enumerate(configs_to_plot_with_names):
        pdr_series = results[config_key]['pdr_series'] * 100 # 转换为百分比
        # 仅绘制到有数据为止的轮次
        valid_rounds = np.where(results[config_key]['alive_series'] > 0)[0]
        if len(valid_rounds) > 0:
            plt.plot(valid_rounds, pdr_series[valid_rounds], label=config_display_name, color=colors[i], linewidth=2)
        else:
            plt.plot([], [], label=config_display_name, color=colors[i], linewidth=2) # 绘制空线以显示图例
    plt.title('PDR 随轮次变化', fontsize=12, fontproperties='SimHei')
    plt.xlabel('轮次', fontsize=10, fontproperties='SimHei')
    plt.ylabel('PDR (%)', fontsize=10, fontproperties='SimHei')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8, prop={'family': 'SimHei'})


    # 剩余能量折线图
    plt.subplot(1, 3, 2)
    for i, (config_key, config_display_name) in enumerate(configs_to_plot_with_names):
        energy_series = results[config_key]['energy_series']
        valid_rounds = np.where(results[config_key]['alive_series'] > 0)[0]
        if len(valid_rounds) > 0:
            plt.plot(valid_rounds, energy_series[valid_rounds], label=config_display_name, color=colors[i], linewidth=2)
        else:
            plt.plot([], [], label=config_display_name, color=colors[i], linewidth=2)
    plt.title('网络剩余能量随轮次变化', fontsize=12, fontproperties='SimHei')
    plt.xlabel('轮次', fontsize=10, fontproperties='SimHei')
    plt.ylabel('剩余能量 (J)', fontsize=10, fontproperties='SimHei')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8, prop={'family': 'SimHei'})

    # 存活节点数折线图
    plt.subplot(1, 3, 3)
    for i, (config_key, config_display_name) in enumerate(configs_to_plot_with_names):
        alive_series = results[config_key]['alive_series']
        valid_rounds = np.where(results[config_key]['alive_series'] > 0)[0]
        if len(valid_rounds) > 0:
            plt.plot(valid_rounds, alive_series[valid_rounds], label=config_display_name, color=colors[i], linewidth=2)
        else:
            plt.plot([], [], label=config_display_name, color=colors[i], linewidth=2)
    plt.title('存活节点数随轮次变化', fontsize=12, fontproperties='SimHei')
    plt.xlabel('轮次', fontsize=10, fontproperties='SimHei')
    plt.ylabel('存活节点数', fontsize=10, fontproperties='SimHei')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8, prop={'family': 'SimHei'})

    plt.tight_layout()
    plt.savefig('weight_experiment_line_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("权重实验折线图已保存为：weight_experiment_line_plots.png")

def print_results_table(results):
    """打印实验结果表格"""
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置颜色方案
    header_color = '#FFF2CC'
    row_colors = ['#FFF2CC', '#D9EAD3']  # 交替行颜色
    text_color = '#000000'
    edge_color = '#CCCCCC'  # 表格边框颜色

    # 创建图形和坐标轴，并隐藏坐标轴
    fig, ax = plt.subplots(figsize=(12, 8))  # 调整图形大小以适应内容
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据和表头
    table_data = []
    headers = ['权重配置', '吞吐量(bps)', '时延(s)', 'PDR(%)', '剩余能量(J)', '存活节点数']

    # 映射权重配置的显示名称
    configs_map = {
        'original': '(0.4,0.3,0.3)',
        'distance': '(0.3,0.4,0.3)',
        'energy': '(0.3,0.3,0.4)',
        'buffer': '(0.6,0.2,0.2)',
        'hybrid': '(0.5,0.3,0.2)', 
        'fuzzy': '(0.4,0.4,0.2)'
    }

    for config_key, config_display_name in configs_map.items():
        r = results[config_key]
        table_data.append([
            config_display_name,
            f"{r['throughput']:.4f}",
            f"{r['delay']:.4f}",
            f"{r['pdr'] * 100:.4f}",  # PDR 以百分比显示
            f"{r['energy']:.4f}",
            f"{r['alive']:.0f}"  # 存活节点数显示为整数
        ])

    # 创建表格
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     loc='center',
                     cellLoc='center')

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)  # 调整表格的宽度和高度
    # 设置标题行和数据行的颜色及文本属性
    for (i, j), cell in table.cells.items():
        if i == 0:  # 表头行
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold', color=text_color)
        else:  # 数据行
            cell.set_facecolor(row_colors[(i - 1) % 2])
            cell.set_text_props(color=text_color)
        cell.set_edgecolor(edge_color)

    # 设置表格标题
    plt.title('不同权重配置下的网络性能对比', fontsize=14, pad=20, fontproperties='SimHei')
    plt.tight_layout()

    # 保存表格图片
    plt.savefig('weight_experiment_results_table.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("权重实验结果表格已保存为：weight_experiment_results_table.png")

def simulate_sensor_data(node, r):
    base_value = random.uniform(0, 1)  # 原有逻辑
    noise = np.random.normal(0, 0.5)  # 均值0，方差0.5的高斯噪声
    return base_value + noise


def test_fault_tolerance_with_protocols():
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    protocols = [
        ('improve_LEACH', run_improve_leach),
        ('LEACH', run_leach),
        ('HEED', run_heed),
        ('D2CRP', run_d2crp)
    ]
    failure_round = 500
    recovery_threshold = 0.9  # PDR恢复到故障前90%
    drop_threshold = 0.8  # 故障判定PDR低于80%
    results = {}
    rounds = np.arange(rmax)

    for proto_name, proto_func in protocols:
        print(f"\n=== 测试协议: {proto_name} ===")
        # 运行协议，获取节点和PDR等
        # 先正常运行到failure_round，获取失效前PDR
        result = proto_func()
        if len(result) >= 9:
            alive, re, pdr, delay, energy_std, throughput, false_wakeup_count_list, false_wakeup_total, false_wakeup_energy = result[:9]
        elif len(result) >= 6:
            alive, re, pdr, delay, energy_std, throughput = result[:6]
            false_wakeup_count_list = [0] * len(alive)
            false_wakeup_total = 0
            false_wakeup_energy = 0
        else:
            raise ValueError("协议返回值数量不匹配")
        pre_failure_pdr = np.mean(pdr[max(0, failure_round - 10):failure_round])
        # 重新运行，注入节点失效
        Node = initialize_network()
        recovery_round = None
        recovery_started = False
        failure_detected_round = None
        lost_packets = 0
        total_packets = 0
        recovery_packets = 0
        recovery_total = 0
        path_switch_count = 0
        main_path_history = {}
        pdr_list = []
        drop_rate_list = []
        recovery_rate_list = []
        for r in range(rmax):
            # 节点失效注入
            if r == failure_round:
                # 选一个存活的节点 (不分角色)
                alive_nodes_indices = [i for i in range(n) if Node[i].flag != 0]
                if alive_nodes_indices:
                    failed_node = random.choice(alive_nodes_indices)
                else:
                    failed_node = None # 没有活着的节点可以失效

                if failed_node is not None:
                    Node[failed_node].flag = 0
                    Node[failed_node].power = 0
                    print(f"{proto_name} 第{failure_round}轮，失效节点: {failed_node}")
                else:
                    print(f"{proto_name} 第{failure_round}轮，没有存活节点可供失效")
            # 路由与PDR统计
            round_packets_sent = 0
            for i in range(n):
                if Node[i].flag != 0:
                    if hasattr(Node[i], 'type') and Node[i].type == 'C':
                        round_packets_sent += len(getattr(Node[i], 'cluster_members', [])) + 1
                    elif hasattr(Node[i], 'CH') and Node[i].CH != -1:
                        round_packets_sent += 1
            # PDR
            if proto_name == 'improve_LEACH':
                # 路径切换统计
                # 需调用forward_through_gateways和bellman_ford_multiobjective
                # 这里只能模拟，假设每次主路径变化都算一次切换
                # 实际应在forward_through_gateways中统计
                # 这里简化为每10轮有一次主路径切换（示意）
                if r > failure_round and r % 10 == 0:
                    path_switch_count += 1
            # 计算PDR
            if proto_name == 'improve_LEACH':
                pdr_val = calculate_round_pdr(Node, sink, round_packets_sent)
            elif proto_name == 'LEACH':
                pdr_val = calculate_leach_pdr(Node, sink, round_packets_sent)
            elif proto_name == 'HEED':
                pdr_val = calculate_heed_pdr(Node, sink, round_packets_sent)
            elif proto_name == 'D2CRP':
                pdr_val = pdr[r] if r < len(pdr) else 1
            elif proto_name == 'TEEN':
                pdr_val = pdr[r] if r < len(pdr) else 1
            else:
                pdr_val = 1
            pdr_list.append(pdr_val)
            # 故障期间丢包率
            if r >= failure_round and r < failure_round + 20:
                if round_packets_sent > 0:
                    drop_rate_list.append(1 - pdr_val)
                    lost_packets += (1 - pdr_val) * round_packets_sent
                    total_packets += round_packets_sent
                else:
                    # 没有数据产生，不计入丢包率统计
                    drop_rate_list.append(np.nan)
            # 恢复检测
            if r > failure_round and not recovery_started and pdr_val < drop_threshold * pre_failure_pdr:
                recovery_started = True
                failure_detected_round = r
            if recovery_started and pdr_val >= recovery_threshold * pre_failure_pdr:
                recovery_round = r
                recovery_started = False
            # 恢复后统计
            if recovery_round and r >= recovery_round and r < recovery_round + 20:
                recovery_packets += pdr_val * round_packets_sent
                recovery_total += round_packets_sent
        # 统计指标
        avg_recovery_time = (recovery_round - failure_detected_round) if (
                    recovery_round and failure_detected_round) else None
        avg_drop_rate = np.mean(drop_rate_list) if drop_rate_list else 0
        recovery_rate = (recovery_packets / recovery_total) if recovery_total > 0 else 0
        results[proto_name] = {
            'recovery_time': avg_recovery_time,
            'drop_rate': avg_drop_rate,
            'recovery_rate': recovery_rate,
            'path_switch_count': path_switch_count
        }
        # 保存PDR曲线
        results[proto_name]['pdr_curve'] = pdr_list
        results[proto_name]['failure_round'] = failure_round
        results[proto_name]['recovery_round'] = recovery_round

   

  
    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        results.keys(),
        [results[k]['drop_rate'] * 100 for k in results],
        color=[(16/255,70/255,128/255), (49/255,124/255,183/255),(109/255,173/255,209/255), (182/255,215/255,232/255),(110/255,207/255,242/255)],
        width=0.4  # 这里设置柱子宽度为0.5，默认是0.8
    )
    plt.ylabel('故障期间丢包率（%）', fontproperties='SimHei')
    plt.title('不同协议故障期间数据丢失率', fontproperties='SimHei')
    plt.tight_layout()
    plt.show()



def detect_delay_anomaly(node, current_delay):
    """使用滑动窗口检测时延异常"""
    if not hasattr(node, 'delay_window'):
        node.delay_window = []
    
    node.delay_window.append(current_delay)
    if len(node.delay_window) > DELAY_WINDOW_SIZE:
        node.delay_window.pop(0)
    
    if len(node.delay_window) < 3:  
        return False
        
    mean_delay = sum(node.delay_window) / len(node.delay_window)
    variance = sum((d - mean_delay) ** 2 for d in node.delay_window) / len(node.delay_window)
    std_dev = variance ** 0.5
    
    # 检测异常：当前时延超过均值+2倍标准差
    return current_delay > mean_delay + DELAY_THRESHOLD_FACTOR * std_dev

def adaptive_local_reroute(node, sink, Node, initial_hop_range=2):
    """自适应局部重路由"""
    hop_range = initial_hop_range
    found_better_path = False
    
    while not found_better_path and hop_range <= MAX_REROUTE_HOPS:
        # 获取hop_range跳范围内的所有节点
        local_nodes = get_nodes_within_hops(node, Node, hop_range)
        
        # 在局部范围内执行Bellman-Ford
        path_tables = bellman_ford_multiobjective(local_nodes, sink, Node)
        
        # 检查是否找到更好的路径
        if path_tables[node] and path_tables[node][0].delay_sum < node.current_path.delay_sum:
            found_better_path = True
            node.current_path = path_tables[node][0]
        else:
            hop_range += 1
            
    return found_better_path

def parallel_forward(gateway, path_tables, sink, Node):
    """并行多路径转发"""
    if not path_tables[gateway]:
        return
            
    # 计算网关节点的负载水平
    load_level = len(gateway.data_buffer) / gateway.buffer_size
    
    if load_level > PARALLEL_PATH_THRESHOLD and len(path_tables[gateway]) > 1:
        # 使用多条路径并行转发
        paths = path_tables[gateway][:3]  # 最多使用3条路径
        
        # 根据路径质量分配数据包
        total_score = sum(1/p.delay_sum for p in paths)
        for path in paths:
            path_ratio = (1/path.delay_sum) / total_score
            packets_for_path = int(len(gateway.data_buffer) * path_ratio)
            
            # 通过该路径转发相应比例的数据包
            for _ in range(packets_for_path):
                if gateway.data_buffer:
                    packet = gateway.data_buffer.pop(0)
                    forward_packet(gateway, path.path, packet, Node)
    else:
        # 负载较轻时仍使用单一最优路径
        best_path = path_tables[gateway][0]
        while gateway.data_buffer:
            packet = gateway.data_buffer.pop(0)
            forward_packet(gateway, best_path.path, packet, Node)

def forward_packet(source, path, packet, Node):
    """沿指定路径转发单个数据包"""
    for i in range(len(path)-1):
        current_node = path[i]
        next_node = path[i+1]
        
        # 计算并消耗转发能量
        dist = calculate_distance(current_node, next_node)
        ETx, ERx = calculate_energy_consumption(current_node, packetLength, dist)
        current_node.power -= ETx  # 发送能量
        
        # 接收节点消耗能量
        if isinstance(next_node, SensorNode):
            next_node.power -= ERx  # 接收能量

def get_nodes_within_hops(start_node, Node, max_hops):
    """获取指定跳数范围内的所有节点"""
    local_nodes = set([start_node])
    current_layer = set([start_node])
    
    for hop in range(max_hops):
        next_layer = set()
        for node in current_layer:
            for neighbor_idx in node.N:
                neighbor = Node[neighbor_idx]
                if neighbor.flag != 0 and neighbor not in local_nodes:
                    next_layer.add(neighbor)
        local_nodes.update(next_layer)
        current_layer = next_layer
        
    return list(local_nodes)

def mac_scheduling_decision(node, current_time, mode):
    """MAC调度决策函数
    
    Args:
        node: 当前节点
        current_time: 当前时间
        mode: MAC调度方式 ('TDMA', 'CSMA/CA', '混合调度', 'IMPROVE_LEACH')
        
    Returns:
        bool: 是否可以在当前时刻发送数据
    """
    if mode == 'TDMA':
        return tdma_schedule(node, current_time)
    elif mode == 'CSMA/CA':
        return perform_csma_ca(node)[0]
    elif mode == '混合调度':
        # 根据数据重要性选择调度方式
        importance = compute_data_importance(node, node.current_value, current_time)
        if importance > 0.7:  # 高重要性数据使用TDMA
            return tdma_schedule(node, current_time)
        else:  # 低重要性数据使用CSMA/CA
            return perform_csma_ca(node)[0]
    else:  # IMPROVE_LEACH原有调度
        return improve_leach_schedule(node, current_time)

def tdma_schedule(node, current_time):
    """TDMA调度函数
    
    Args:
        node: 当前节点
        current_time: 当前时间（轮次）
        
    Returns:
        bool: 是否可以在当前时隙发送数据
    """
    if node.type == 'C':  # 簇头节点总是可以发送
        return True
        
    if not hasattr(node, 'tdma_slot') or node.tdma_slot is None:
        return False
        
    # 计算当前时间在FRAME内的相对位置
    frame_time = current_time % FRAME_TIME
    return node.tdma_slot['start'] <= frame_time < node.tdma_slot['end']

def csma_ca_schedule(node):
    """CSMA/CA调度"""
    if not hasattr(node, 'backoff_counter'):
        node.backoff_counter = 0
    
    # 使用已有的MAX_BACKOFF参数
    while node.backoff_counter < MAX_BACKOFF:
        # 执行CSMA/CA
        if perform_csma_ca(node)[0]:
            node.backoff_counter = 0
            return True
        node.backoff_counter += 1
    
    node.backoff_counter = 0
    return False

def perform_csma_ca(node, max_backoff=3, failure_strategy='switch'):
    """
    执行CSMA/CA,支持可配置的最大重传次数和失败处理策略
    Args:
        node: 节点对象
        max_backoff: 最大重传次数,默认为3
        failure_strategy: 失败处理策略 ('switch' 或 'drop')
    Returns:
        tuple: (是否传输成功, 失败处理建议)
    """
    backoff_count = 0
    while backoff_count < max_backoff:
        # 执行退避
        cw = min(pow(2, backoff_count) * CW_MIN, CW_MAX) 
        backoff_slots = random.randint(0, cw-1)
        
        # 模拟信道检测
        channel_busy = random.random() < 0.3  # 30%概率信道忙
        if not channel_busy:
            return True, None
        
        backoff_count += 1
        time.sleep(SLOT_TIME * backoff_slots)  # 模拟退避时延
    
    # 达到最大退避次数后的处理
    if failure_strategy == 'switch':
        return False, 'switch_path'  # 建议切换路径
    elif failure_strategy == 'drop':
        return False, 'drop_packet'  # 建议丢弃数据包
    elif failure_strategy == 'ch_to_sink':
        if hasattr(node, 'type') and node.type == 'C':
            return False, 'connect_sink'  # 簇头直连基站
        else:
            return False, 'drop_packet'  # 簇成员丢包
    else:
        return False, None

def run_backoff_comparison():
    """
    对比不同重传次数(3-7)对网络性能的影响
    """
    print("开始退避重传次数对比实验...")
    
    # 1. 减小实验规模
    global n, rmax
    n = 100   
    rounds = 3  
    
    # 2. 关闭图形显示
    global is_display
    is_display = False
    
    # 3. 添加简单的传感器数据模拟
    def simulate_sensor_data(node, r):
        return random.uniform(0, 1)
    
    # 4. 其他代码保持不变
    backoff_attempts = range(3, 8)
    results = {
        'delays': [],
        'pdrs': []
    }
    
    for max_backoff in backoff_attempts:
        print(f"\n测试重传次数: {max_backoff}")
        delays = []
        pdrs = []
        
        try:
            Node = initialize_network()
            for r in range(rounds):
                print(f"\n当前进度: {r+1}/{rounds}")
                
                global MAX_BACKOFF
                MAX_BACKOFF = max_backoff
                
                result = run_improve_leach(
                    config={
                        'max_backoff': max_backoff,
                        'num_nodes': n,
                        'round': r
                    }
                )
                
                if result:
                    _, _, pdr, delay, _, _, _, _, _ = result
                    delays.append(np.mean(delay[delay > 0]))
                    pdrs.append(np.mean(pdr[pdr > 0]))
            
            print(f"\n完成重传次数{max_backoff}的测试")
            if delays and pdrs:
                avg_delay = np.mean(delays)
                avg_pdr = np.mean(pdrs)
                results['delays'].append(avg_delay)
                results['pdrs'].append(avg_pdr)
                print(f"平均时延: {avg_delay:.4f}")
                print(f"平均PDR: {avg_pdr:.4f}")
            
        except Exception as e:
            print(f"\n测试出错: {str(e)}")
            continue
    
    if results['delays'] and results['pdrs']:
        plot_backoff_comparison(backoff_attempts, results)
        save_backoff_results(backoff_attempts, results)
    
    return results  # 添加这行，返回结果字典

def plot_backoff_comparison(backoff_attempts, results):
    """
    绘制重传次数对比图表
    """
    plt.figure(figsize=(12, 6))
    
    # 设置双Y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 绘制时延曲线
    line1 = ax1.plot(backoff_attempts, results['delays'], color='#A4C2F4', 
                     label='时延', marker='o', markersize=8, linewidth=2)
    ax1.set_xlabel('最大重传次数')
    ax1.set_ylabel('平均时延(ms)')
    
    # 绘制PDR曲线
    line2 = ax2.plot(backoff_attempts, results['pdrs'], color='#B6D7A8', 
                     label='PDR', marker='s', markersize=8, linewidth=2)
    ax2.set_ylabel('PDR')
    
    # 设置图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper right')
    
    plt.title('不同重传次数的时延和PDR对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴刻度
    plt.xticks(backoff_attempts)
    
    plt.tight_layout()
    plt.savefig('backoff_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_backoff_results(backoff_attempts, results):
    """
    保存实验结果到表格
    """
    # 创建结果表格
    plt.figure(figsize=(8, 6))
    table_data = []
    headers = ['组合编号', '最大重传次数', '平均时延(ms)', 'PDR']
    
    for i, max_backoff in enumerate(backoff_attempts, 1):
        table_data.append([
            i,
            max_backoff,
            f"{results['delays'][i-1]:.4f}",
            f"{results['pdrs'][i-1]:.4f}"
        ])
    
    # 创建表格
    table = plt.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#FFF2CC'] * len(headers),
        cellColours=[['#FFF2CC' if i % 2 == 0 else '#D9EAD3' 
                     for _ in range(len(headers))] 
                    for i in range(len(table_data))]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 隐藏坐标轴
    plt.axis('off')
    
    plt.title('重传次数实验结果表')
    plt.tight_layout()
    
    # 保存表格
    plt.savefig('backoff_results_table.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

def improve_leach_schedule(node, current_time):
    """improve_leach原有的调度方式"""
    # 使用improve_leach中原有的调度逻辑
    if node.type == 'N':
        return tdma_schedule(node, current_time)
    else:
        return True  # 簇头节点始终可以发送

def run_mac_comparison():
    """运行MAC调度方式对比实验"""
    global n, rmax
    
    results = {mode: {'delay': [], 'pdr': [], 'energy': []} 
              for mode in MAC_MODE.values()}
    
    for mode in MAC_MODE.values():
        print(f"\n测试{mode}调度方式...")
        Node = initialize_network()
        sink = BaseStation()
        
        for r in range(rmax):
            # 重置节点状态
            for i in range(n):
                Node[i].type = 'N'
                Node[i].selected = 'N'
                Node[i].temp_rand = random.random()
                Node[i].Rc = calculate_communication_radius(Node[i])
                Node[i].CH = 0
                Node[i].is_gateway = False
                Node[i].parent = None
                Node[i].cluster_members = []
                Node[i].layer = calculate_node_layer(Node[i], sink)
                
                # 邻居节点
                Node[i].N = []
                for j in range(n):
                    if i != j and Node[j].flag != 0:
                        dist = calculate_distance(Node[i], Node[j])
                        if dist < Node[i].Rc:
                            Node[i].N.append(j)
                Node[i].Num_N = len(Node[i].N)
            
            # 构建路由表
            build_routing_table(Node, sink)
            
            # 簇头选举
            candidate_CH = []
            potential_CH = []
            
            # 第一阶段：通过阈值筛选潜在候选
            for i in range(n):
                if Node[i].flag != 0 and not Node[i].is_gateway:
                    threshold = (p) / (1 - p * (r % round(1 / p)))
                    if Node[i].temp_rand <= threshold:
                        potential_CH.append(i)
            
            # 第二阶段：应用模糊逻辑选择簇头
            if potential_CH:
                max_energy = max(Node[i].power for i in potential_CH)
                chance_values = {}
                for i in potential_CH:
                    chance_values[i] = calculate_fuzzy_chance(Node[i], Node, max_energy)
                
                optimal_ch_count = max(int(len([n for n in Node if n.flag != 0]) * 0.1), 1)
                selected_CH = sorted(chance_values.items(), key=lambda x: x[1], reverse=True)[:optimal_ch_count]
                
                for ch_id, _ in selected_CH:
                    if perform_csma_ca(Node[ch_id])[0]:
                        Node[ch_id].type = 'C'
                        Node[ch_id].selected = 'O'
                        Node[ch_id].CH = -1
                        Node[ch_id].Rc = Rmax * Node[ch_id].power / E0
                        candidate_CH.append(ch_id)
                        
                        broadcast_energy = calculate_energy_consumption(
                            Node[ch_id],
                            ctrPacketLength,
                            Node[ch_id].Rc,
                            is_control=True
                        )
                        Node[ch_id].power -= broadcast_energy
            
            # 簇成员加入和TDMA时隙分配
            for i in range(n):
                if Node[i].flag != 0 and Node[i].type == 'N':
                    min_dist = float('inf')
                    selected_CH = -1
                    for ch in candidate_CH:
                        dist = calculate_distance(Node[i], Node[ch])
                        if dist < Node[ch].Rc and dist < min_dist:
                            min_dist = dist
                            selected_CH = ch
                    
                    if selected_CH != -1:
                        join_energy = calculate_energy_consumption(
                            Node[i],
                            ctrPacketLength,
                            min_dist,
                            is_control=True
                        )
                        Node[i].power -= join_energy
                        Node[selected_CH].power -= Eelec * ctrPacketLength
                        
                        Node[i].CH = selected_CH
                        Node[selected_CH].cluster_members.append(i)
                        
                        # 分配TDMA时隙
                        if Node[selected_CH].cluster_members:
                            time_slots = assign_tdma_slot(Node[selected_CH], Node[selected_CH].cluster_members)
                            for member_idx, slot_info in time_slots.items():
                                Node[member_idx].tdma_slot = slot_info
            
            # 数据传输阶段
            round_stats = {
                'delay': 0,
                'packets_sent': 0,
                'packets_received': 0,
                'energy': 0
            }
            
            for i in range(n):
                if Node[i].flag == 0:
                    continue
                    
                if mac_scheduling_decision(Node[i], r, mode):
                    target = Node[Node[i].CH] if Node[i].CH != -1 else sink
                    dist = calculate_distance(Node[i], target)
                    
                    ETx, ERx = calculate_energy_consumption(Node[i], packetLength, dist)
                    round_stats['energy'] += ETx + ERx  # 总能量消耗
                    Node[i].power -= ETx  # 发送能量
                    if isinstance(target, SensorNode):
                        target.power -= ERx  # 接收能量
                    
                    delay = calculate_delay(Node[i], sink, packetLength, Node)
                    round_stats['delay'] += delay
                    round_stats['packets_sent'] += 1
                    
                    if random.random() < get_link_quality(Node[i], target):
                        round_stats['packets_received'] += 1
            
            if round_stats['packets_sent'] > 0:
                results[mode]['delay'].append(round_stats['delay'] / round_stats['packets_sent'])
                results[mode]['pdr'].append(round_stats['packets_received'] / round_stats['packets_sent'])
                results[mode]['energy'].append(round_stats['energy'] / n)
    
    return results

def plot_mac_comparison_results(results):
    """绘制MAC协议对比结果"""
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 设置颜色方案
    colors = ['#EFFCEE', '#FEC7E4', '#D4EEFF', '#FFE5D6']
    protocols = ['TDMA', 'CSMA_CA', 'HYBRID', 'FDMMBF-ARP']
    
    # 平均时延对比
    bar_width = 0.2
    x = np.arange(len(protocols))
    delays = [np.mean(results[p]['delay']) for p in protocols]  # 改为'delay'
    ax1.bar(x, delays, width=bar_width, color=colors)
    ax1.set_ylabel('平均时延(s)', fontproperties='SimHei')
    ax1.set_title('平均时延对比', fontproperties='SimHei')
    ax1.set_xticks(x)
    ax1.set_xticklabels(protocols, rotation=45)
    
    # 数据包传输成功率对比
    pdrs = [np.mean(results[p]['pdr']) for p in protocols]  # 改为'pdr'
    ax2.bar(x, pdrs, width=bar_width, color=colors)
    ax2.set_ylabel('PDR', fontproperties='SimHei')
    ax2.set_title('数据包传输成功率对比', fontproperties='SimHei')
    ax2.set_xticks(x)
    ax2.set_xticklabels(protocols, rotation=45)
    
    # 节点平均能耗对比
    energy = [np.mean(results[p]['energy']) for p in protocols]
    ax3.bar(x, energy, width=bar_width, color=colors)
    ax3.set_ylabel('能耗(J)', fontproperties='SimHei')
    ax3.set_title('节点平均能耗对比', fontproperties='SimHei')
    ax3.set_xticks(x)
    ax3.set_xticklabels(protocols, rotation=45)
    
    plt.tight_layout()
    # 设置白色背景
    fig.patch.set_facecolor('white')
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('white')
    
    plt.savefig('mac_comparison_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def display_network(Node, sink, r):
    """显示网络拓扑"""
    pass

def plot_protocol_comparison(alive_nodes_improve_leach, alive_nodes_leach, alive_nodes_heed, alive_nodes_d2crp,
                           residual_energy_improve_leach, residual_energy_leach, residual_energy_heed, residual_energy_d2crp):
    """绘制协议比较结果"""
    plt.figure(figsize=(15, 10))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 存活节点数量对比
    plt.subplot(2, 3, 1)
    for protocol in PROTOCOL_COLORS.keys():
        data = eval(f'alive_nodes_{protocol.lower()}')
        plt.plot(data, label=protocol, color=PROTOCOL_COLORS[protocol], linewidth=2)
    plt.title('节点存活数量对比', fontsize=12)
    plt.xlabel('轮次', fontsize=10)
    plt.ylabel('存活节点数', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=9)
    
    # 剩余能量对比
    plt.subplot(2, 3, 2)
    for protocol in PROTOCOL_COLORS.keys():
        data = eval(f'residual_energy_{protocol.lower()}')
        plt.plot(data, label=protocol, color=PROTOCOL_COLORS[protocol], linewidth=2)
    plt.title('网络剩余能量对比', fontsize=12)
    plt.xlabel('轮次', fontsize=10)
    plt.ylabel('剩余能量(J)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=9)
    
    # 吞吐量对比
    plt.subplot(2, 3, 3)
    for protocol in PROTOCOL_COLORS.keys():
        data = eval(f'throughput_{protocol.lower()}')
        plt.plot(data, label=protocol, color=PROTOCOL_COLORS[protocol], linewidth=2)
    plt.title('网络吞吐量对比', fontsize=12)
    plt.xlabel('轮次', fontsize=10)
    plt.ylabel('吞吐量(bps)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=9)
    
    # 时延对比
    plt.subplot(2, 3, 4)
    for protocol in PROTOCOL_COLORS.keys():
        data = eval(f'delay_{protocol.lower()}')
        plt.plot(data, label=protocol, color=PROTOCOL_COLORS[protocol], linewidth=2)
    plt.title('网络时延对比', fontsize=12)
    plt.xlabel('轮次', fontsize=10)
    plt.ylabel('时延(s)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=9)
    
    # PDR对比
    plt.subplot(2, 3, 5)
    for protocol in PROTOCOL_COLORS.keys():
        data = eval(f'pdr_{protocol.lower()}')
        plt.plot(data, label=protocol, color=PROTOCOL_COLORS[protocol], linewidth=2)
    plt.title('数据包传输成功率对比', fontsize=12)
    plt.xlabel('轮次', fontsize=10)
    plt.ylabel('PDR(%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('protocol_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_failure_strategies():
    """
    比较两种失败处理策略的性能表现
    1. 切换策略：连续退避3次失败后切换路径
    2. 丢包策略：连续退避3次失败后直接丢弃
    3. 簇头直连基站/簇成员丢包：连续退避3次失败后簇头直连基站，簇成员丢包
    """
    print("\n=== 失败处理策略对比实验 ===")
    print("实验设置:")
    print("- 每种策略运行3次实验")
    print("- 每次实验运行100轮")
    print("- 连续退避3次失败后触发策略")
    print("- 记录平均传输时延和PDR\n")
    # 实验参数设置
    num_rounds = 100
    num_experiments = 1
    
    # 存储实验结果
    switch_delays = []
    drop_delays = []
    ch_to_sink_delays = []
    switch_pdrs = []
    drop_pdrs = []
    ch_to_sink_pdrs = []
    
    for exp in range(num_experiments):
        print(f"\n=== 运行实验 {exp + 1}/{num_experiments} ===")
        
        # 切换策略实验
        print("\n1. 测试切换策略:")
        Node, sink = initialize_network()
        config = {
            'failure_strategy': 'switch',
            'max_backoff_failures': 5
        }
        _, _, pdr_switch, delay_switch, _, _, _, _, _ = run_improve_leach(config)
        switch_delays.extend(delay_switch)
        switch_pdrs.extend(pdr_switch)
        print(f"- 平均时延: {np.mean(delay_switch):.4f}ms")
        print(f"- PDR: {np.mean(pdr_switch):.4f}%")
        
        # 丢包策略实验
        print("\n2. 测试丢包策略:")
        Node, sink = initialize_network()
        config = {
            'failure_strategy': 'drop',
            'max_backoff_failures': 5
        }
        _, _, pdr_drop, delay_drop, _, _, _, _, _ = run_improve_leach(config)
        drop_delays.extend(delay_drop)
        drop_pdrs.extend(pdr_drop)
        print(f"- 平均时延: {np.mean(delay_drop):.4f}ms")
        print(f"- PDR: {np.mean(pdr_drop):.4f}%")


        # 新策略实验
        print("\n3. 测试簇头直连基站/簇成员丢包策略:")
        Node, sink = initialize_network()
        config = {
            'failure_strategy': 'ch_to_sink',
            'max_backoff_failures': 5
        }
        _, _, pdr_ch_to_sink, delay_ch_to_sink, _, _, _, _, _ = run_improve_leach(config)
        ch_to_sink_delays.extend(delay_ch_to_sink)
        ch_to_sink_pdrs.extend(pdr_ch_to_sink)
        print(f"- 平均时延: {np.mean(delay_ch_to_sink):.4f}ms")
        print(f"- PDR: {np.mean(pdr_ch_to_sink):.4f}%")
    
    # 计算平均值
    avg_switch_delay = np.mean(switch_delays)
    avg_drop_delay = np.mean(drop_delays)
    avg_ch_to_sink_delay = np.mean(ch_to_sink_delays)
    avg_switch_pdr = np.mean(switch_pdrs)
    avg_drop_pdr = np.mean(drop_pdrs)
    avg_ch_to_sink_pdr = np.mean(ch_to_sink_pdrs)
    
    # 绘制结果柱状图
    plt.figure(figsize=(12, 5))
    # 延迟对比
    plt.subplot(1, 2, 1)
    bars = plt.bar(['切换策略', '丢包策略', '簇头直连基站/簇成员丢包'],
                  [avg_switch_delay, avg_drop_delay, avg_ch_to_sink_delay],
                  color=['#EFFCEE', '#D4EEFF', '#FFE5D6'],
                  width=0.6)
    plt.title('平均传输时延对比')
    plt.ylabel('时延 (ms)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    # PDR对比
    plt.subplot(1, 2, 2)
    bars = plt.bar(['切换策略', '丢包策略', '簇头直连基站/簇成员丢包'],
                  [avg_switch_pdr, avg_drop_pdr, avg_ch_to_sink_pdr],
                  color=['#EFFCEE', '#D4EEFF', '#FFE5D6'],
                  width=0.6)
    plt.title('数据包传输成功率对比')
    plt.ylabel('PDR (%)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}%',
                ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
    # 打印详细结果
    print("\n实验结果统计:")
    print(f"切换策略 - 平均时延: {avg_switch_delay:.4f}ms, PDR: {avg_switch_pdr:.4f}%")
    print(f"丢包策略 - 平均时延: {avg_drop_delay:.4f}ms, PDR: {avg_drop_pdr:.4f}%")
    print(f"簇头直连基站/簇成员丢包 - 平均时延: {avg_ch_to_sink_delay:.4f}ms, PDR: {avg_ch_to_sink_pdr:.4f}%")
    return avg_switch_delay, avg_switch_pdr, avg_drop_delay, avg_drop_pdr, avg_ch_to_sink_delay, avg_ch_to_sink_pdr

def run_pdr_vs_nodes_experiment():
    global n, rmax, is_display
    original_n = n
    original_rmax = rmax
    original_is_display = is_display

    is_display = False # Disable dynamic display for experiments

    node_counts = [ 20, 30, 40, 50, 60, 70, 80, 90,100] # Range of node numbers to test, adjusted to match the provided image
    # For this specific comparison, we will fix rmax to a single value for consistency with the example graph.
    # Using 3000 rounds for the comparison.
    fixed_rmax = 1000

    # Store PDR results for each protocol
    all_protocols_pdr_results = {
        'FDMMBF-ARP': [],
        'LEACH': [],
        'HEED': [],
        'D2CRP': []
    }

    protocols_to_run = {
        'FDMMBF-ARP': run_improve_leach,
        'LEACH': run_leach,
        'HEED': run_heed,
        'D2CRP': run_d2crp
    }

    try:
        rmax = fixed_rmax # Set global rmax for this experiment
        print(f"\n--- 运行协议比较实验 (固定轮次 rmax = {rmax}) ---")
        
        for proto_name, proto_func in protocols_to_run.items():
            print(f"  正在测试协议: {proto_name}")
            current_protocol_pdr_values = []
            for node_count in node_counts:
                n = node_count # Set global n
                print(f"    测试节点数: {n}...")
                
                # Run the specific protocol function
                result = proto_func()
                
                # Extract PDR series based on the result structure
                # Assuming PDR is the 3rd element (index 2) for all these functions
                pdr_series = result[2]
                
                # Calculate average PDR for the current node_count
                valid_pdr_values = [val for val in pdr_series if val > 0 and not np.isnan(val)]
                # Convert to percentage and cap at 100
                avg_pdr = min(np.mean(valid_pdr_values) * 100, 100) if valid_pdr_values else 0 
                current_protocol_pdr_values.append(avg_pdr)
                print(f"      平均 PDR: {avg_pdr:.2f}%")
            all_protocols_pdr_results[proto_name] = current_protocol_pdr_values

        # Plotting the results
        plot_pdr_vs_nodes_results_combined(node_counts, all_protocols_pdr_results)

    finally:
        # Restore original global variables
        n = original_n
        rmax = original_rmax
        is_display = original_is_display
        print("\n实验完成，全局变量已恢复。")

def plot_pdr_vs_nodes_results_combined(node_counts, all_protocols_pdr_results):
    plt.figure(figsize=(14, 8)) # Adjusted figure size to be similar to the provided image
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    original_font_family = plt.rcParams['font.family']
    original_font_size = plt.rcParams['font.size']
    
    # 为英文图表设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False

    # Define colors and markers to match the provided image as closely as possible
    plot_styles = {
        'LEACH': {'color': '#FF9800', 'marker': 'o', 'label': 'LEACH'}, 
        'HEED': {'color': '#388E3C', 'marker': 'o', 'label': 'HEED'}, 
        'D2CRP': {'color': '#9B59B6', 'marker': 's', 'label': 'D2CRP'}, 
        'FDMMBF-ARP': {'color': '#C0392B', 'marker': '^', 'label': 'FD-MFAR'}, 
    }
  
    sorted_protocols = ['FDMMBF-ARP', 'LEACH', 'HEED', 'D2CRP'] # Reorder to match potential legend order if needed

    for protocol in sorted_protocols:
        if protocol in all_protocols_pdr_results:
            style = plot_styles.get(protocol, {'color': 'black', 'marker': 'o', 'label': protocol}) # Default style if not found
            # Ensure the data for plotting is not empty and lengths match
            pdr_data = all_protocols_pdr_results[protocol]
            if len(pdr_data) > 0 and len(node_counts) == len(pdr_data):
                # 将PDR数据除以100，使其与Y轴0-1的范围匹配
                plt.plot(node_counts, [p / 100 for p in pdr_data], 
                         marker=style['marker'], linestyle='-', color=style['color'], 
                         linewidth=2, label=style['label'])
            else:
                print(f"警告: 协议 {protocol} 的数据不完整或为空，无法绘制曲线。Node counts: {len(node_counts)}, PDR data: {len(pdr_data)}")
        

    plt.title('Scheduling Algorithm Performance Comparison', fontsize=16,fontfamily='Times New Roman') # Main title from the image
    plt.xlabel('Number of Nodes', fontsize=14,fontfamily='Times New Roman') # X-axis label from the image
    plt.ylabel('Packet Delivery Rate', fontsize=14,fontfamily='Times New Roman') # Y-axis label from the image (Acceptance Rate)
    plt.grid(True) # Ensure grid is visible
    plt.ylim(0.15, 0.85) # Y-axis limits from the image
    plt.xticks(node_counts,fontfamily='Times New Roman') # Changed to use actual node_counts for ticks
    plt.yticks(np.arange(0.1, 0.9, 0.1),fontfamily='Times New Roman') # Y-axis ticks from the image
    plt.legend(loc='upper left', fontsize=10,prop={'family': 'Times New Roman'}) # Adjust legend location and font size

    plt.tight_layout()
    plt.savefig('protocol_pdr_vs_nodes_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("数据包传输成功率对比图已保存为：protocol_pdr_vs_nodes_comparison.png")

# === 修复后的函数：在特定节点数下运行协议 PDR 随轮次变化实验 ===
def run_pdr_over_rounds_for_single_node_count(current_n_value):
    global n, rmax, is_display
    original_n = n
    original_rmax = 1000
    original_is_display = is_display
    n = current_n_value # 设置全局节点数 n
    is_display = False # 禁用动态显示

    rmax = 1000 # 将实验轮数设置为 1000 轮

    protocols_to_run = {
        'FDMMBF-ARP': run_improve_leach,
        'LEACH': run_leach,
        'HEED': run_heed,
        'D2CRP': run_d2crp
    }

    pdr_results_for_protocols = {}

    print(f"\n--- 运行节点数为 {current_n_value} 时的协议 PDR 随轮次变化实验 (固定轮次 rmax = {rmax}) ---")
    
    for proto_name, proto_func in protocols_to_run.items():
        print(f"  正在测试协议: {proto_name} (节点数: {current_n_value})...")
        
        result = proto_func() 
        
        # PDR 序列通常是返回值的第三个元素（索引为 2）
        pdr_series = result[2] 
        
        # Debug: 打印每个协议的原始PDR序列信息
        print(f"    DEBUG: {proto_name}原始PDR序列长度: {len(pdr_series)}")
        print(f"    DEBUG: {proto_name}原始PDR序列类型: {type(pdr_series)}")
        
        # 检查数据中NaN的分布
        nan_count = sum(1 for val in pdr_series if np.isnan(val))
        print(f"    DEBUG: {proto_name}中NaN值数量: {nan_count}/{len(pdr_series)}")
        
        # 修改过滤逻辑：保持原始序列长度，用前一个有效值替换NaN
        processed_pdr = []
        last_valid_value = 0.0  # 默认初始值
        
        for i, val in enumerate(pdr_series):
            if np.isnan(val):
                # 如果是NaN，使用最后一个有效值，如果没有有效值则使用0
                processed_pdr.append(last_valid_value * 100)
                if i < 10 or i % 100 == 0:  # 只在前10个和每100轮打印一次调试信息
                    print(f"    DEBUG: {proto_name}轮次{i+1}: NaN -> {last_valid_value * 100:.2f}%")
            else:
                processed_pdr.append(val * 100)
                last_valid_value = val
        
        pdr_results_for_protocols[proto_name] = processed_pdr
        
        # Debug: 打印处理后的数据长度
        print(f"    DEBUG: {proto_name}处理后PDR数据长度: {len(pdr_results_for_protocols[proto_name])}")
        
        # 检查600轮后的数据
        if len(processed_pdr) > 600:
            post_600_data = processed_pdr[600:700]  # 检查600-700轮的数据
            post_600_avg = np.mean(post_600_data) if post_600_data else 0
            print(f"    DEBUG: {proto_name}第600-700轮平均PDR: {post_600_avg:.2f}%")
        
        if pdr_results_for_protocols[proto_name]:
            print(f"    平均 PDR ({proto_name}, 节点数 {current_n_value}): {np.mean(pdr_results_for_protocols[proto_name]):.2f}%")
        else:
            print(f"    警告: 协议 {proto_name} 在节点数 {current_n_value} 时没有有效的 PDR 数据。")

    n = original_n
    rmax = original_rmax
    is_display = original_is_display
    print(f"--- 节点数为 {current_n_value} 的协议 PDR 实验完成 ---")
    
    return pdr_results_for_protocols

# === 修复后的绘图函数：PDR 随轮次变化图 ===
def plot_pdr_over_rounds_comparison(pdr_data_all_protocols, current_n_value):
    plt.figure(figsize=(14, 8))  # 增大图形尺寸
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    original_font_family = plt.rcParams['font.family']
    original_font_size = plt.rcParams['font.size']
    
    # 为英文图表设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False

    # 定义颜色和标记以匹配提供的图表格式
    plot_styles = {
        'FDMMBF-ARP': {'color': '#C0392B', 'marker': '^', 'label': 'FD-MFAR', 'markersize': 6},  # 紫色
        'LEACH': {'color': '#FF9800', 'marker': 'o', 'label': 'LEACH', 'markersize': 6},              # 橙色
        'HEED': {'color': '#388E3C', 'marker': 'o', 'label': 'HEED', 'markersize': 6},                # 绿色
        'D2CRP': {'color': '#9B59B6', 'marker': 's', 'label': 'D2CRP', 'markersize': 6},              # 红色
    }

    # 排序协议以匹配图例顺序
    sorted_protocols_for_plot = ['FDMMBF-ARP', 'LEACH', 'HEED', 'D2CRP']

    # 确保所有协议数据长度一致
    min_length = float('inf')
    for proto_name in sorted_protocols_for_plot:
        if proto_name in pdr_data_all_protocols and pdr_data_all_protocols[proto_name]:
            min_length = min(min_length, len(pdr_data_all_protocols[proto_name]))
    
    if min_length == float('inf'):
        print("错误: 没有有效的协议数据")
        return
    
    print(f"DEBUG: 所有协议的最小数据长度: {min_length}")

    # 使用固定的1000轮作为最大轮数
    max_rounds_to_plot = min(1000, min_length)
    plot_interval = 100  # 每100轮一个数据点

    for protocol in sorted_protocols_for_plot:
        if protocol in pdr_data_all_protocols and pdr_data_all_protocols[protocol]:
            style = plot_styles.get(protocol, {'color': 'black', 'marker': 'o', 'label': protocol, 'markersize': 6})
            pdr_values_raw = pdr_data_all_protocols[protocol][:max_rounds_to_plot] # 截取固定长度
            
            print(f"DEBUG: {protocol}用于绘图的数据长度: {len(pdr_values_raw)}")

            # 计算每100轮的平均PDR
            averaged_pdr_data = []
            rounds_for_plotting = []
            
            for i in range(0, len(pdr_values_raw), plot_interval):
                chunk = pdr_values_raw[i:i+plot_interval]
                if chunk: # 确保块不为空
                    # 数据已经是0-100范围，转换为0-1范围
                    chunk_normalized = [p / 100 for p in chunk]
                    avg_pdr = np.mean(chunk_normalized)
                    averaged_pdr_data.append(avg_pdr)
                    rounds_for_plotting.append(i + plot_interval) # 使用100轮区间的结束点
                    
                    # Debug: 打印关键数据点
                    if (i + plot_interval) in [100, 200, 600, 700, 1000]:
                        print(f"DEBUG: {protocol}第{i+1}-{i+plot_interval}轮平均PDR: {avg_pdr:.4f}")

            if averaged_pdr_data and rounds_for_plotting:
                print(f"DEBUG: {protocol}绘图数据点数量: {len(averaged_pdr_data)}")
                plt.plot(rounds_for_plotting, averaged_pdr_data,
                         marker=style['marker'], linestyle='-', color=style['color'],
                         linewidth=2, label=style['label'], markersize=style['markersize'],
                         markerfacecolor=style['color'], markeredgecolor=style['color'])
            else:
                print(f"警告: 协议 {protocol} 在节点数 {current_n_value} 时没有足够的有效数据点进行绘制。")
        else:
            print(f"警告: 协议 {protocol} 在节点数 {current_n_value} 时没有有效的 PDR 数据，无法绘制曲线。")

    plt.title(f'Scheduling Algorithm Performance Comparison (Node Count={current_n_value})', fontsize=16,fontfamily='Times New Roman')
    plt.xlabel('Round', fontsize=14,fontfamily='Times New Roman')
    plt.ylabel('Packet Delivery Rate', fontsize=14,fontfamily='Times New Roman')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1) # Y 轴范围调整为 0 到 1.05
    plt.yticks(np.arange(0, 1.1, 0.1)) # Y 轴刻度从 0 到 1，增量为 0.1

    # 设置 X 轴刻度为 100, 200, ..., 1000
    plt.xticks(np.arange(100, 1001, 100))
    plt.xlim(50, 1050)  # 给X轴留一些边距

    plt.legend(loc='lower left', fontsize=10,prop={'family': 'Times New Roman'})

    plt.tight_layout()
    
    # 保存高质量图片
    filename = f'protocol_pdr_vs_rounds_n{current_n_value}_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"修复后的节点数为 {current_n_value} 的数据包传输成功率对比图已保存为：{filename}")

# === 额外的诊断函数：检查协议数据完整性 ===
def diagnose_protocol_data(pdr_data_all_protocols, current_n_value):
    """诊断各协议数据的完整性"""
    print(f"\n=== 协议数据诊断 (节点数={current_n_value}) ===")
    
    for protocol, data in pdr_data_all_protocols.items():
        if data:
            print(f"\n{protocol}:")
            print(f"  数据长度: {len(data)}")
            print(f"  数据范围: {min(data):.2f}% - {max(data):.2f}%")
            print(f"  平均值: {np.mean(data):.2f}%")
            
            # 检查不同轮次段的数据
            segments = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
            for start, end in segments:
                if len(data) > start:
                    segment_data = data[start:min(end, len(data))]
                    if segment_data:
                        avg_val = np.mean(segment_data)
                        print(f"  第{start+1}-{min(end, len(data))}轮平均: {avg_val:.2f}%")
        else:
            print(f"\n{protocol}: 无数据")
    
    print("=== 诊断完成 ===\n")

def run_energy_comparison_experiment():
    """
    运行FDMMBF-ARP、leach、heed、d2crp协议的剩余能量对比实验。
    """
    global n, rmax, is_display
    original_n = n
    original_rmax = rmax
    original_is_display = is_display

    n = 400 # 实验节点数可以根据需要调整
    rmax = 3000 # 实验轮数可以根据需要调整
    is_display = False # 关闭动态显示以加速实验

    print(f"\n--- 运行剩余能量对比实验 (节点数={n}, 轮次={rmax}) ---")

    # 运行各个协议并收集数据
    print("正在运行 FDMMBF-ARP 协议...")
    alive_nodes_improve_leach, residual_energy_improve_leach = run_improve_leach()[:2]
    
    print("正在运行 LEACH 协议...")
    alive_nodes_leach, residual_energy_leach = run_leach()[:2]
    
    print("正在运行 HEED 协议...")
    alive_nodes_heed, residual_energy_heed = run_heed()[:2]
    
    print("正在运行 D2CRP 协议...")
    alive_nodes_d2crp, residual_energy_d2crp = run_d2crp()[:2]

    print("实验数据收集完成，正在生成对比图...")

    # 调用绘图函数
    plot_alive_nodes_comparison(
        alive_nodes_improve_leach,
        alive_nodes_leach,
        alive_nodes_heed,
        alive_nodes_d2crp
    )
    
    plot_residual_energy_comparison(
        residual_energy_improve_leach,
        residual_energy_leach,
        residual_energy_heed,
        residual_energy_d2crp
    )

    # 恢复全局变量
    n = original_n
    rmax = original_rmax
    is_display = original_is_display
    print("\n剩余能量对比实验完成，全局变量已恢复。")




def run_lifetime_comparison_experiment():
    """
    对比FDMMBF-ARP、LEACH、HEED、D2CRP协议的网络整体寿命（FND/HND/LND）。
    """
    global n, is_display
    original_n = n
    original_is_display = is_display

    n = 400
    is_display = False

    original_font_family = plt.rcParams['font.family']
    original_font_size = plt.rcParams['font.size']
    
    # 为英文图表设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    print(f"\n--- 运行网络寿命对比实验 (节点数={n}) ---")
    print("注意：每个协议将运行到所有节点死亡为止，可能需要较长时间...")

    def run_protocol_until_death(protocol_name, protocol_func):
        """运行协议直到所有节点死亡"""
        print(f"正在运行 {protocol_name} 协议...")
        
        # 临时设置一个很大的rmax值，确保能运行到所有节点死亡
        global rmax, E0
        original_rmax = rmax
        original_E0 = E0

        if protocol_name == "FDMMBF-ARP":
            E0 = 0.3
        else:
            E0 = 0.5

        rmax = 20000  # 设置一个很大的值
        
        try:
            # 运行协议
            if protocol_name == "FDMMBF-ARP":
                alive_data, _, _, _, _, _, _, _, _ = protocol_func()
            else:
                alive_data, _, _, _, _, _ = protocol_func()
            
            # 找到最后一个非零值的索引（即最后一个节点死亡的轮数）
            last_alive_index = len(alive_data) - 1
            while last_alive_index >= 0 and alive_data[last_alive_index] == 0:
                last_alive_index -= 1
            
            # 截取到最后一个节点死亡的轮数
            if last_alive_index >= 0:
                alive_data = alive_data[:last_alive_index + 1]
            
            print(f"{protocol_name} 协议完成，运行了 {len(alive_data)} 轮")
            return alive_data
            
        finally:
            rmax = original_rmax
            E0 = original_E0  # 恢复原始rmax值

    # 运行各协议，收集每轮存活节点数
    alive_fdmm = run_protocol_until_death("FDMMBF-ARP", run_improve_leach)
    alive_leach = run_protocol_until_death("LEACH", run_leach)
    alive_heed = run_protocol_until_death("HEED", run_heed)
    alive_d2crp = run_protocol_until_death("D2CRP", run_d2crp)

    # 计算FND/HND/LND
    def calc_lifetime(alive_arr):
        total = alive_arr[0]  # 初始节点数
        
        # 找到最后一个非零值的索引
        last_nonzero = len(alive_arr) - 1
        while last_nonzero >= 0 and alive_arr[last_nonzero] == 0:
            last_nonzero -= 1
        
        # 如果所有值都是0，说明协议没有运行
        if last_nonzero < 0:
            return 0, 0, 0
        
        # 截取到最后一个非零值
        alive_arr = alive_arr[:last_nonzero + 1]
        
        fnd = next((i for i, v in enumerate(alive_arr) if v < total), len(alive_arr))
        hnd = next((i for i, v in enumerate(alive_arr) if v <= total/2), len(alive_arr))
        lnd = next((i for i, v in enumerate(alive_arr) if v == 0), len(alive_arr))
        return fnd, hnd, lnd

    fnd_fdmm, hnd_fdmm, lnd_fdmm = calc_lifetime(alive_fdmm)
    fnd_leach, hnd_leach, lnd_leach = calc_lifetime(alive_leach)
    fnd_heed, hnd_heed, lnd_heed = calc_lifetime(alive_heed)
    fnd_d2crp, hnd_d2crp, lnd_d2crp = calc_lifetime(alive_d2crp)

    # 打印详细结果
    print("\n=== 网络寿命对比结果 ===")
    print(f"{'协议':<12} {'FND':<8} {'HND':<8} {'LND':<8} {'总轮数':<8}")
    print("-" * 50)
    print(f"{'FDMMBF-ARP':<12} {fnd_fdmm:<8} {hnd_fdmm:<8} {lnd_fdmm:<8} {len(alive_fdmm):<8}")
    print(f"{'LEACH':<12} {fnd_leach:<8} {hnd_leach:<8} {lnd_leach:<8} {len(alive_leach):<8}")
    print(f"{'HEED':<12} {fnd_heed:<8} {hnd_heed:<8} {lnd_heed:<8} {len(alive_heed):<8}")
    print(f"{'D2CRP':<12} {fnd_d2crp:<8} {hnd_d2crp:<8} {lnd_d2crp:<8} {len(alive_d2crp):<8}")

    # 绘制对比柱状图
    protocols = ['FD-MFAR', 'LEACH', 'HEED', 'D2CRP']
    FND = [fnd_fdmm, fnd_leach, fnd_heed, fnd_d2crp]
    HND = [hnd_fdmm, hnd_leach, hnd_heed, hnd_d2crp]
    LND = [lnd_fdmm, lnd_leach, lnd_heed, lnd_d2crp]

    x = np.arange(len(protocols))
    width = 0.25
    
    # 统一颜色方案
    color_fnd = '#60B8B5'  # FND统一使用青色
    color_hnd = '#78B7C9'  # HND统一使用蓝色
    color_lnd = '#FF8C00'  # LND统一使用橙色
    
    plt.figure(figsize=(14, 8))
    
    # 绘制FND和HND
    plt.bar(x-width, FND, width, label='FND (First Node Death)', color=color_fnd, alpha=0.8, edgecolor='black', linewidth=1)
    plt.bar(x, HND, width, label='HND (50% Node Death)', color=color_hnd, alpha=0.8, edgecolor='black', linewidth=1)

    # LND处理 - 现在所有协议都运行到全部死亡
    bars = plt.bar(x+width, LND, width, label='LND (All Nodes Death)', 
                   color=color_lnd, alpha=0.8, edgecolor='black', linewidth=1)

    # 在柱顶添加数值标签
    for i, (fnd, hnd, lnd) in enumerate(zip(FND, HND, LND)):
        plt.text(x[i]-width, fnd+max(FND)*0.01, str(fnd), ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(x[i], hnd+max(HND)*0.01, str(hnd), ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(x[i]+width, lnd+max(LND)*0.01, str(lnd), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(x, protocols, fontsize=12, fontfamily='Times New Roman')
    plt.ylabel('rounds', fontsize=12, fontfamily='Times New Roman')
    plt.title('Network Lifetime Comparison\n(Run until all nodes die)', fontsize=14, fontfamily='Times New Roman')
    plt.legend(fontsize=12, loc='upper left', prop={'family':'Times New Roman'})
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 添加数值表格
    table_data = []
    alive_data_list = [alive_fdmm, alive_leach, alive_heed, alive_d2crp]
    for i, protocol in enumerate(protocols):
        table_data.append([protocol, FND[i], HND[i], LND[i], len(alive_data_list[i])])
    
    # 在图表下方添加表格
    table = plt.table(cellText=table_data,
                     colLabels=['Protocol', 'FND', 'HND', 'LND', 'Total Rounds'],
                     cellLoc='center',
                     loc='bottom',
                     bbox=[0, -0.15, 1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 为表格留出空间
    plt.savefig('lifetime_comparison_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制存活节点数随时间变化曲线
    plt.figure(figsize=(14, 8))
    
    # 确定最大轮数
    max_rounds = max(len(alive_fdmm), len(alive_leach), len(alive_heed), len(alive_d2crp))
    
    # 创建时间轴
    rounds = np.arange(max_rounds)
    
    # 绘制存活节点数曲线
    plt.plot(rounds[:len(alive_fdmm)], alive_fdmm, 'o-', label='FD-MFAR', linewidth=2, markersize=4, color=color_fnd)
    plt.plot(rounds[:len(alive_leach)], alive_leach, 's-', label='LEACH', linewidth=2, markersize=4, color=color_hnd)
    plt.plot(rounds[:len(alive_heed)], alive_heed, '^-', label='HEED', linewidth=2, markersize=4, color=color_lnd)
    plt.plot(rounds[:len(alive_d2crp)], alive_d2crp, 'd-', label='D2CRP', linewidth=2, markersize=4, color='#1f77b4')
    
    plt.xlabel('rounds', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Alive Nodes', fontsize=14, fontweight='bold')
    plt.title('Comparison of Alive Nodes for Different Protocols', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('alive_nodes_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 恢复全局变量
    n = original_n
    is_display = original_is_display
    print("\n网络寿命对比实验完成，全局变量已恢复。")

def run_energy_balance_experiment():
    """
    对比FDMMBF-ARP、LEACH、HEED、D2CRP协议的能量均衡性（每轮活跃节点能量标准差）。
    """
    global n, rmax, is_display
    original_n = n
    original_rmax = rmax
    original_is_display = is_display
    # 保存原始字体设置
    original_font_family = plt.rcParams['font.family']
    original_font_size = plt.rcParams['font.size']
    
    # 为英文图表设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    n = 200
    rmax = 1000
    is_display = False

    print(f"\n--- 运行能量均衡性对比实验 (节点数={n}, 轮次={rmax}) ---")

    # 运行各协议，收集每轮能量标准差
    print("正在运行 FDMMBF-ARP 协议...")
    std_fdmm = run_improve_leach()[4]  # 第5个返回值为energy_std_data
    print("正在运行 LEACH 协议...")
    std_leach = run_leach()[-3]
    print("正在运行 HEED 协议...")
    std_heed = run_heed()[-3]
    print("正在运行 D2CRP 协议...")
    std_d2crp = run_d2crp()[-3]

    # 转换为numpy数组
    std_fdmm = np.array(std_fdmm, dtype=float)
    std_leach = np.array(std_leach, dtype=float)
    std_heed = np.array(std_heed, dtype=float)
    std_d2crp = np.array(std_d2crp, dtype=float)

    # 打印调试信息
    print("\n调试信息:")
    print(f"FDMMBF-ARP 能量标准差范围: {np.min(std_fdmm):.6f} - {np.max(std_fdmm):.6f}")
    print(f"LEACH 能量标准差范围: {np.min(std_leach):.6f} - {np.max(std_leach):.6f}")
    print(f"HEED 能量标准差范围: {np.min(std_heed):.6f} - {np.max(std_heed):.6f}")
    print(f"D2CRP 能量标准差范围: {np.min(std_d2crp):.6f} - {np.max(std_d2crp):.6f}")

    # 数据预处理：将过小的值（接近0）替换为nan
    min_threshold = 1e-6
    std_fdmm = np.where(std_fdmm < min_threshold, np.nan, std_fdmm)
    std_leach = np.where(std_leach < min_threshold, np.nan, std_leach)
    std_heed = np.where(std_heed < min_threshold, np.nan, std_heed)
    std_d2crp = np.where(std_d2crp < min_threshold, np.nan, std_d2crp)
    
    # 将数据分成5个阶段，计算每个阶段的平均值
    stages = 5
    stage_size = rmax // stages
    
    # 初始化存储每个阶段的平均值
    fdmm_stages = []
    leach_stages = []
    heed_stages = []
    d2crp_stages = []
    
    for i in range(stages):
        start_idx = i * stage_size
        end_idx = (i + 1) * stage_size
        
        fdmm_stages.append(np.nanmean(std_fdmm[start_idx:end_idx]))
        leach_stages.append(np.nanmean(std_leach[start_idx:end_idx]))
        heed_stages.append(np.nanmean(std_heed[start_idx:end_idx]))
        d2crp_stages.append(np.nanmean(std_d2crp[start_idx:end_idx]))
    
    # 创建柱状图
    plt.figure(figsize=(14, 8))
    
    # 设置柱状图的位置
    x = np.arange(stages)
    width = 0.2  # 柱子的宽度
    
    # 绘制柱状图
    plt.bar(x - width*1.5, fdmm_stages, width, label='FD-MFAR', color='#E58B7B', alpha=0.8, edgecolor='black', linewidth=1)
    plt.bar(x - width/2, leach_stages, width, label='LEACH', color='#60B8B5', alpha=0.8, edgecolor='black', linewidth=1)
    plt.bar(x + width/2, heed_stages, width, label='HEED', color='#78B7C9', alpha=0.8, edgecolor='black', linewidth=1)
    plt.bar(x + width*1.5, d2crp_stages, width, label='D2CRP', color='#FF8C00', alpha=0.8, edgecolor='black', linewidth=1)
    
    # 设置图表属性
    plt.xlabel('Operation Phase', fontsize=12,fontfamily = 'Times New Roman')
    plt.ylabel('Average Energy Standard Deviation', fontsize=12,fontfamily = 'Times New Roman')
    plt.title('Energy Balance Comparison in Different Phases', fontsize=14,fontfamily = 'Times New Roman')
    
    # 设置x轴刻度
    plt.xticks(x, [f'Phase {i+1}\n({i*stage_size}-{(i+1)*stage_size} rounds)' for i in range(stages)], fontsize=10,fontfamily = 'Times New Roman')
    plt.yticks(fontsize=10,fontfamily = 'Times New Roman')
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加图例
    plt.legend(fontsize=10, loc='upper left',prop = {'family':'Times New Roman'})
    
    plt.tight_layout()
    plt.savefig('energy_balance_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印各阶段的具体数值
    print("\n各阶段平均能量标准差:")
    for i in range(stages):
        print(f"\n第{i+1}阶段 ({i*stage_size}-{(i+1)*stage_size}轮):")
        print(f"FDMMBF-ARP: {fdmm_stages[i]:.6f}")
        print(f"LEACH: {leach_stages[i]:.6f}")
        print(f"HEED: {heed_stages[i]:.6f}")
        print(f"D2CRP: {d2crp_stages[i]:.6f}")

    # 恢复全局变量
    n = original_n
    rmax = original_rmax
    is_display = original_is_display
    
    # 恢复原始字体设置
    plt.rcParams['font.family'] = original_font_family
    plt.rcParams['font.size'] = original_font_size
    
    print("\n能量均衡性对比实验完成，全局变量已恢复。")

def smooth_data(data, window_size=5):
    """使用移动平均平滑数据"""
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed

def plot_alive_nodes_comparison(alive_nodes_improve_leach, alive_nodes_leach, alive_nodes_heed, alive_nodes_d2crp):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(14, 8))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.unicode_minus'] = False
    
    original_font_family = plt.rcParams['font.family']
    original_font_size = plt.rcParams['font.size']
    
    # 为英文图表设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    
    rounds = range(len(alive_nodes_improve_leach))
    
    plot_styles = {
        'FD-MFAR': {'color': '#C0392B', 'marker': '^', 'label': 'FD-MFAR'},
        'LEACH': {'color': '#FF9800', 'marker': 'o', 'label': 'LEACH'},
        'HEED': {'color': '#388E3C', 'marker': 'o', 'label': 'HEED'},
        'D2CRP': {'color': '#9B59B6', 'marker': 's', 'label': 'D2CRP'},
    }
    
    protocol_data = [
        ('FD-MFAR', smooth_data(alive_nodes_improve_leach, window_size=80)),
        ('LEACH', smooth_data(alive_nodes_leach, window_size=80)),
        ('HEED', smooth_data(alive_nodes_heed, window_size=80)),
        ('D2CRP', smooth_data(alive_nodes_d2crp, window_size=80)),
    ]
    
    for protocol, data in protocol_data:
        style = plot_styles[protocol]
        if len(data) > 0 and len(rounds) == len(data):
            plt.plot(rounds, data, marker=style['marker'], linestyle='-', color=style['color'], linewidth=2, label=style['label'], markersize=8, markevery=200)
        else:
            print(f"警告: 协议 {protocol} 的数据不完整或为空，无法绘制曲线。Rounds: {len(rounds)}, Data: {len(data)}")
    
    plt.xlabel('Rounds', fontsize=14,fontfamily='Times New Roman')
    plt.ylabel('Number of Alive Nodes', fontsize=14,fontfamily='Times New Roman')
    plt.title('Comparison of Alive Nodes for Different Protocols', fontsize=16,fontfamily='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7)
    # 修改x轴刻度，确保显示到3000轮
    max_rounds = max(len(alive_nodes_improve_leach), len(alive_nodes_leach), 
                     len(alive_nodes_heed), len(alive_nodes_d2crp))
    plt.xticks(range(0, max_rounds + 1, 300))
    plt.xlim(0, max_rounds)
    plt.legend(loc='upper right', fontsize=10,prop={'family': 'Times New Roman'})
    plt.tight_layout()
    plt.show()

def plot_residual_energy_comparison(residual_energy_improve_leach, residual_energy_leach, residual_energy_heed, residual_energy_d2crp):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(14, 8))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.unicode_minus'] = False
    original_font_family = plt.rcParams['font.family']
    original_font_size = plt.rcParams['font.size']
    
    # 为英文图表设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    
    rounds = range(len(residual_energy_improve_leach))
    plot_styles = {
        'FD-MFAR': {'color': '#C0392B', 'marker': '^', 'label': 'FD-MFAR'},
        'LEACH': {'color': '#FF9800', 'marker': 'o', 'label': 'LEACH'},
        'HEED': {'color': '#388E3C', 'marker': 'o', 'label': 'HEED'},
        'D2CRP': {'color': '#9B59B6', 'marker': 's', 'label': 'D2CRP'},
    }
    protocol_data = [
        ('FD-MFAR', residual_energy_improve_leach),
        ('LEACH', residual_energy_leach),
        ('HEED', residual_energy_heed),
        ('D2CRP', residual_energy_d2crp),
    ]
    for protocol, data in protocol_data:
        style = plot_styles[protocol]
        if len(data) > 0 and len(rounds) == len(data):
            plt.plot(rounds, data, marker=style['marker'], linestyle='-', color=style['color'], linewidth=2, label=style['label'], markersize=8, markevery=200)
        else:
            print(f"警告: 协议 {protocol} 的数据不完整或为空，无法绘制曲线。Rounds: {len(rounds)}, Data: {len(data)}")
    plt.xlabel('Rounds', fontsize=14,fontfamily='Times New Roman')
    plt.ylabel('Remaining Energy (J)', fontsize=14,fontfamily='Times New Roman')
    plt.title('Comparison of Remaining Energy for Different Protocols', fontsize=16,fontfamily='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7)
    # 修改x轴刻度，确保显示到3000轮
    max_rounds = max(len(residual_energy_improve_leach), len(residual_energy_leach), 
                     len(residual_energy_heed), len(residual_energy_d2crp))
    plt.xticks(range(0, max_rounds + 1, 300))
    plt.xlim(0, max_rounds)
    plt.legend(loc='upper right', fontsize=10,prop={'family': 'Times New Roman'})
    plt.tight_layout()
    plt.show()


# 如果直接运行此文件，则执行对比实验
if __name__ == "__main__":
    # 实验控制开关
    RUN_PROTOCOL_COMPARISON = False    # 运行协议比较实验 (旧的 PDR vs 节点数)
    RUN_MAC_COMPARISON = False       # 运行MAC调度方式对比实验
    RUN_BACKOFF_COMPARISON = False    # 运行退避重传次数对比实验
    enable_weight_experiments = False  # 权重实验开关
    RUN_FAILURE_STRATEGY_COMPARISON = False 
    RUN_PDR_VS_NODES_EXPERIMENT = False  # 禁用旧的 PDR vs 节点数实验

    # === 新增：PDR 随轮次变化实验控制开关 ===
    RUN_PDR_OVER_ROUNDS_BY_NODES = False 

    # === 新增：剩余能量、存活节点数、吞吐量对比实验控制开关 ===
    RUN_ENERGY_COMPARISON_EXPERIMENT = False

    RUN_LIFETIME_COMPARISON_EXPERIMENT = False  # 新增：网络寿命对比实验开关

    RUN_ENERGY_BALANCE_EXPERIMENT = False # Add this line to define the variable


    if RUN_FAILURE_STRATEGY_COMPARISON:
        print("\n开始运行失败处理策略对比实验...")
        compare_failure_strategies()
        print("失败处理策略对比实验完成！")

    if RUN_BACKOFF_COMPARISON:
        print("\n=== 运行退避重传次数对比实验 ===")
        results = run_backoff_comparison()
        backoff_attempts = range(3, 8)
        plot_backoff_comparison(backoff_attempts, results)
        print("\n实验结果已保存")

    if RUN_MAC_COMPARISON:
        print("\n=== 运行MAC调度方式对比实验 ===")
        results = run_mac_comparison()
        plot_mac_comparison_results(results)
        
        print("\n各调度方式平均性能指标：")
        for mode in MAC_MODE.values():
            avg_delay = sum(results[mode]['delay']) / len(results[mode]['delay'])
            avg_pdr = sum(results[mode]['pdr']) / len(results[mode]['pdr'])
            avg_energy = sum(results[mode]['energy']) / len(results[mode]['energy'])
            
            print(f"\n{mode}调度方式:")
            print(f"平均端到端时延: {avg_delay*1000:.2f}ms")
            print(f"平均PDR: {avg_pdr*100:.2f}%")
            print(f"平均节点能耗: {avg_energy:.6f}J")
        
        print("\n实验结果图已保存为：mac_comparison_results.png")

    if RUN_PROTOCOL_COMPARISON:
        print("\n=== 运行协议比较实验 (RUN_PROTOCOL_COMPARISON = True) ===")
        run_pdr_vs_nodes_experiment() # Call the modified function
        print("协议比较实验完成！")

    if enable_weight_experiments:
        print("\n开始运行权重实验...")
        results = run_weight_experiments()
        print_results_table(results)
        plot_weight_experiment_line_charts(results)
        print("权重实验图表已生成")

    if RUN_PDR_VS_NODES_EXPERIMENT:
        print("\n开始运行数据包传输成功率 vs 节点数实验...")
        run_pdr_vs_nodes_experiment()
        print("数据包传输成功率 vs 节点数实验完成！")
    
    if RUN_PDR_OVER_ROUNDS_BY_NODES:
        print("\n=== 运行 PDR 随轮次变化实验 (按节点数分组) ===")
        node_counts_for_rounds_experiment = [350, 400, 450]
        for current_n in node_counts_for_rounds_experiment:
            pdr_data_for_current_n = run_pdr_over_rounds_for_single_node_count(current_n)
            plot_pdr_over_rounds_comparison(pdr_data_for_current_n, current_n)
        print("PDR 随轮次变化实验完成！")
    
    if RUN_ENERGY_COMPARISON_EXPERIMENT:
        print("\n=== 运行剩余能量对比实验 ===")
        run_energy_comparison_experiment()
        print("剩余能量对比实验完成！")

    if RUN_LIFETIME_COMPARISON_EXPERIMENT:
        print("\n=== 运行网络寿命对比实验 ===")
        run_lifetime_comparison_experiment()
        print("网络寿命对比实验完成！")

    if RUN_ENERGY_BALANCE_EXPERIMENT:
        print("\n=== 运行能量均衡性对比实验 ===")
        run_energy_balance_experiment()
        print("能量均衡性对比实验完成！")


