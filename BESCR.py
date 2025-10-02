import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math
from typing import List, Dict, Tuple, Optional
from scipy.signal import savgol_filter  # type: ignore
import skfuzzy as fuzz
from skfuzzy import control as ctrl



class BESCRNode:
    """BESCR算法节点类"""
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 0.6):
        self.id = node_id
        self.x = x
        self.y = y
        self.initial_energy = initial_energy
        self.remaining_energy = initial_energy
        self.is_cluster_head = False
        self.cluster_id = -1
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_dropped = 0
        self.trust_direct = 1.0
        self.trust_indirect = 1.0
        self.trust_combined = 1.0
        self.is_malicious = False
        self.neighbor_trusts = {}
        self.neighbor_count = 0
        
    def update_trust(self, eta: float = 0.7):
        """更新信任值"""
        if self.packets_received > 0:
            self.trust_direct = self.packets_sent / self.packets_received
        else:
            self.trust_direct = 1.0
            
        if self.neighbor_trusts:
            self.trust_indirect = np.mean(list(self.neighbor_trusts.values()))
        else:
            self.trust_indirect = 1.0
            
        self.trust_combined = eta * self.trust_direct + (1 - eta) * self.trust_indirect
        
    def consume_energy(self, energy: float):
        self.remaining_energy = max(0, self.remaining_energy - energy)
        
    def is_alive(self) -> bool:
        return self.remaining_energy > 0
        
    def get_distance_to(self, other_x: float, other_y: float) -> float:
        return math.sqrt((self.x - other_x)**2 + (self.y - other_y)**2)

class BESCREnergyModel:
    """BESCR能量模型类"""
    def __init__(self):
        self.elec_energy = 50e-9
        self.free_space_energy = 10e-12
        self.multi_path_energy = 0.001e-12
        self.data_aggr_energy = 5e-9
        self.threshold_distance = math.sqrt(self.free_space_energy / self.multi_path_energy)
        
    def calculate_transmit_energy(self, k: int, distance: float) -> float:
        if distance <= self.threshold_distance:
            return k * self.elec_energy + k * self.free_space_energy * (distance**2)
        else:
            return k * self.elec_energy + k * self.multi_path_energy * (distance**4)
            
    def calculate_receive_energy(self, k: int) -> float:
        return k * self.elec_energy
        
    def calculate_aggregation_energy(self, k: int) -> float:
        return k * self.data_aggr_energy

class BESCRCGABCO:
    """BESCR改进的人工蜂群优化算法"""
    def __init__(self, num_nodes: int, num_clusters: int, max_iterations: int = 50, 
                 colony_size: int = 30, mutation_rate: float = 0.05):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.colony_size = colony_size
        self.mutation_rate = mutation_rate
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def logistic_tent_chaos(self, x: float) -> float:
        """Logistic-Tent混沌映射"""
        if x < 0.5:
            return 4 * x * (1 - x)
        else:
            return 4 * (1 - x) * x
            
    def generate_initial_solution(self) -> List[int]:
        """使用混沌映射生成初始解"""
        x = random.random()
        solution = []
        
        for _ in range(self.num_nodes):
            x = self.logistic_tent_chaos(x)
            if random.random() < x:
                solution.append(1)
            else:
                solution.append(0)
                
        # 确保有正确的簇头数量
        current_ch_count = sum(solution)
        if current_ch_count > self.num_clusters:
            ones = [i for i, val in enumerate(solution) if val == 1]
            random.shuffle(ones)
            for i in ones[:current_ch_count - self.num_clusters]:
                solution[i] = 0
        elif current_ch_count < self.num_clusters:
            zeros = [i for i, val in enumerate(solution) if val == 0]
            random.shuffle(zeros)
            for i in zeros[:self.num_clusters - current_ch_count]:
                solution[i] = 1
                
        return solution
        
    def sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))
        
    def update_employee_bee(self, solution: List[int], best_solution: List[int]) -> List[int]:
        """雇佣蜂阶段更新"""
        new_solution = solution.copy()
        
        for i in range(self.num_nodes):
            if random.random() < 0.5:
                r = random.uniform(-1, 1)
                new_value = solution[i] + (solution[i] - best_solution[i]) * r
                sig_value = self.sigmoid(new_value)
                new_solution[i] = 1 if random.random() < sig_value else 0
                
        return new_solution
        
    def apply_genetic_mutation(self, solution: List[int]) -> List[int]:
        """应用遗传变异算子"""
        new_solution = solution.copy()
        
        for i in range(self.num_nodes):
            if random.random() < self.mutation_rate:
                new_solution[i] = 1 - new_solution[i]
                
        # 确保簇头数量正确
        current_ch_count = sum(new_solution)
        if current_ch_count > self.num_clusters:
            ones = [i for i, val in enumerate(new_solution) if val == 1]
            random.shuffle(ones)
            for i in ones[:current_ch_count - self.num_clusters]:
                new_solution[i] = 0
        elif current_ch_count < self.num_clusters:
            zeros = [i for i, val in enumerate(new_solution) if val == 0]
            random.shuffle(zeros)
            for i in zeros[:self.num_clusters - current_ch_count]:
                new_solution[i] = 1
                
        return new_solution
        
    def optimize(self, nodes: List[BESCRNode], bs_x: float, bs_y: float, 
                 energy_model: BESCREnergyModel) -> List[int]:
        """执行优化过程"""
        # 初始化种群
        population = []
        fitness_values = []
        
        for _ in range(self.colony_size):
            solution = self.generate_initial_solution()
            population.append(solution)
            fitness = self.calculate_fitness(solution, nodes, bs_x, bs_y, energy_model)
            fitness_values.append(fitness)
            
        # 找到最优解
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        
        for iteration in range(self.max_iterations):
            # 雇佣蜂阶段
            for i in range(self.colony_size):
                new_solution = self.update_employee_bee(population[i], self.best_solution)
                new_fitness = self.calculate_fitness(new_solution, nodes, bs_x, bs_y, energy_model)
                
                if new_fitness < fitness_values[i]:
                    population[i] = new_solution
                    fitness_values[i] = new_fitness
                    
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_solution.copy()
                        self.best_fitness = new_fitness
                        
            # 观察蜂阶段
            total_fitness = sum(fitness_values)
            probabilities = [f / total_fitness if total_fitness > 0 else 1 / len(fitness_values) 
                           for f in fitness_values]
            
            for _ in range(self.colony_size):
                selected_idx = np.random.choice(len(population), p=probabilities)
                new_solution = self.update_employee_bee(population[selected_idx], self.best_solution)
                new_fitness = self.calculate_fitness(new_solution, nodes, bs_x, bs_y, energy_model)
                
                if new_fitness < fitness_values[selected_idx]:
                    population[selected_idx] = new_solution
                    fitness_values[selected_idx] = new_fitness
                    
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_solution.copy()
                        self.best_fitness = new_fitness
                        
            # 侦查蜂阶段
            for i in range(self.colony_size):
                if random.random() < 0.1:
                    population[i] = self.generate_initial_solution()
                    fitness_values[i] = self.calculate_fitness(population[i], nodes, bs_x, bs_y, energy_model)
                    
            # 遗传变异
            for i in range(self.colony_size):
                population[i] = self.apply_genetic_mutation(population[i])
                fitness_values[i] = self.calculate_fitness(population[i], nodes, bs_x, bs_y, energy_model)
                
                if fitness_values[i] < self.best_fitness:
                    self.best_solution = population[i].copy()
                    self.best_fitness = fitness_values[i]
                    
        return self.best_solution
        
    def calculate_fitness(self, solution: List[int], nodes: List[BESCRNode], 
                         bs_x: float, bs_y: float, energy_model: BESCREnergyModel) -> float:
        """计算适应度函数"""
        if sum(solution) != self.num_clusters:
            return float('inf')
            
        cluster_heads = [i for i, val in enumerate(solution) if val == 1]
        
        # f1: 簇头剩余能量之和（最大化）
        f1 = sum(nodes[i].remaining_energy for i in cluster_heads)
        
        # f2: 簇内总距离（最小化）
        f2 = 0
        for ch_id in cluster_heads:
            ch_node = nodes[ch_id]
            for node in nodes:
                if not node.is_cluster_head and node.cluster_id == ch_id:
                    distance = ch_node.get_distance_to(node.x, node.y)
                    f2 += distance
                    
        # f3: 簇头到BS的总距离（最小化）
        f3 = sum(nodes[i].get_distance_to(bs_x, bs_y) for i in cluster_heads)
        
        # f4: 总延迟（最小化）
        f4 = 0
        chi = 0.4
        v = 50000
        
        for ch_id in cluster_heads:
            ch_node = nodes[ch_id]
            distance_to_bs = ch_node.get_distance_to(bs_x, bs_y)
            k = 4000
            delay = chi * (distance_to_bs**2) + k / v
            f4 += delay
            
        # f5: 簇头信任值之和（最大化）
        f5 = sum(nodes[i].trust_combined for i in cluster_heads)
        
        # 归一化
        f1_norm = f1 / (len(cluster_heads) * 0.6) if cluster_heads else 0
        f2_norm = f2 / (len(nodes) * 100) if f2 > 0 else 1
        f3_norm = f3 / (len(cluster_heads) * 100) if cluster_heads else 1
        f4_norm = f4 / (len(cluster_heads) * 1000) if cluster_heads else 1
        f5_norm = f5 / len(cluster_heads) if cluster_heads else 1
        
        # 权重
        w1, w2, w3, w4, w5 = 0.3, 0.2, 0.2, 0.1, 0.2
        
        # 目标函数
        fitness = (w2 * f2_norm + w3 * f3_norm + w4 * f4_norm) / (w1 * f1_norm + w5 * f5_norm + 1e-6)
        
        return fitness

class BESCRProtocol:
    """BESCR协议实现"""
    def __init__(self, network_size: int = 100, area_size: float = 100.0):
        self.network_size = network_size
        self.area_size = area_size
        self.nodes = []
        self.bs_x = area_size / 2
        self.bs_y = area_size / 2
        self.energy_model = BESCREnergyModel()
        self.current_round = 0
        self.packets_transferred = 0
        self.packets_dropped = 0
        self.alive_nodes = []
        self.dead_nodes = []
        self.network_lifetime_metrics = {'fnd': -1, 'hnd': -1, 'lnd': -1}
        self.alive_nodes_data = []
        self.residual_energy_data = []
        self.energy_std_data = []
        self.pdr_data = []
        
    def initialize_network(self, num_nodes: int = None):
        """初始化网络"""
        if num_nodes is None:
            num_nodes = self.network_size
            
        self.nodes = []
        
        # 随机部署节点
        for i in range(num_nodes):
            x = random.uniform(0, self.area_size)
            y = random.uniform(0, self.area_size)
            node = BESCRNode(i, x, y)
            
            # 随机设置一些节点为恶意节点（5%概率）
            if random.random() < 0.05:
                node.is_malicious = True
                node.trust_direct = 0.3
                node.trust_indirect = 0.3
                node.trust_combined = 0.3
                
            self.nodes.append(node)
            
        self.alive_nodes = self.nodes.copy()
        self.dead_nodes = []
        self.current_round = 0
        self.alive_nodes_data = []
        self.residual_energy_data = []
        self.energy_std_data = []
        self.pdr_data = []
        
    def run_clustering_round(self):
        """运行一轮聚类"""
        self.current_round += 1
        
        # 更新所有节点的信任值
        for node in self.alive_nodes:
            node.update_trust()
            
        # 使用CGABCO选择簇头
        num_clusters = max(1, int(0.1 * len(self.alive_nodes)))
        cgabco = BESCRCGABCO(len(self.alive_nodes), num_clusters)
        
        # 重置簇头状态
        for node in self.alive_nodes:
            node.is_cluster_head = False
            node.cluster_id = -1
            
        # 优化选择簇头
        cluster_solution = cgabco.optimize(self.alive_nodes, self.bs_x, self.bs_y, self.energy_model)
        
        # 设置簇头
        cluster_heads = []
        for i, is_ch in enumerate(cluster_solution):
            if is_ch and i < len(self.alive_nodes):
                self.alive_nodes[i].is_cluster_head = True
                self.alive_nodes[i].cluster_id = i
                cluster_heads.append(self.alive_nodes[i])
                
        # 普通节点加入最近的簇头
        for node in self.alive_nodes:
            if not node.is_cluster_head:
                min_distance = float('inf')
                best_ch = None
                
                for ch in cluster_heads:
                    distance = node.get_distance_to(ch.x, ch.y)
                    if distance < min_distance:
                        min_distance = distance
                        best_ch = ch
                        
                if best_ch:
                    node.cluster_id = best_ch.id
                    
        # 执行数据传输
        self.perform_data_transmission()
        
        # 更新网络统计
        self.update_network_statistics()
        
    # 在BESCRProtocol类中添加链路质量计算函数
    

    def calculate_bescr_pdr(self, round_packets_sent):
        """BESCR协议PDR计算，修复重复计算和定义错误"""
        if round_packets_sent == 0:
            return 0.0
    
        total_successful_packets = 0
        
        for node in self.alive_nodes:
            if node.is_alive() and node.is_cluster_head:
                # 计算到基站的链路质量（更严格的模型）
                distance_to_bs = node.get_distance_to(self.bs_x, self.bs_y)
                if distance_to_bs > 0:
                    # 增加路径损耗，使成功率更现实
                    path_loss = 30 * math.log10(distance_to_bs + 1) + 15 * math.log10(distance_to_bs + 1)
                    snr = max(5, 80 - path_loss)  # 降低基础SNR，最小SNR为5dB
                else:
                    snr = 80
                
                # 基于SNR的链路质量（更保守）
                if snr >= 25:
                    link_quality = 0.75  # 降低最高成功率
                elif snr >= 20:
                    link_quality = 0.60
                elif snr >= 15:
                    link_quality = 0.40
                elif snr >= 10:
                    link_quality = 0.25
                else:
                    link_quality = 0.10  # 增加最低成功率
                
                # 能量状态影响更明显
                energy_status = node.remaining_energy / node.initial_energy
                energy_factor = 0.3 + 0.7 * energy_status  # 能量影响更大
                
                # 恶意节点影响更严重
                malicious_factor = 0.3 if node.is_malicious else 1.0
            
                # 簇头数据聚合成功率（降低）
                cluster_members = [n for n in self.alive_nodes if n.cluster_id == node.id and not n.is_cluster_head]
                aggregation_success = 0.7 if len(cluster_members) > 0 else 1.0
            
                # 综合成功率（更严格）
                success_rate = (link_quality * 0.4 + 
                              energy_factor * 0.4 + 
                              aggregation_success * 0.2) * malicious_factor
              
            
                # 簇头发送的数据包数
                ch_packets = 1 + len(cluster_members)
                total_successful_packets += ch_packets * success_rate
    
        # 计算簇内通信成功率
        for node in self.alive_nodes:
            if node.is_alive() and not node.is_cluster_head and node.cluster_id != -1:
                ch = next((n for n in self.alive_nodes if n.id == node.cluster_id and n.is_cluster_head), None)
                if ch and ch.is_alive():
                    # 计算到簇头的链路质量（更严格）
                    ch_distance = node.get_distance_to(ch.x, ch.y)
                    if ch_distance > 0:
                        ch_path_loss = 25 * math.log10(ch_distance + 1) + 10 * math.log10(ch_distance + 1)
                        ch_snr = max(5, 85 - ch_path_loss)
                    else:
                        ch_snr = 85
                    
                    # 基于SNR的链路质量（更保守）
                    if ch_snr >= 25:
                        ch_link_quality = 0.80
                    elif ch_snr >= 20:
                        ch_link_quality = 0.65
                    elif ch_snr >= 15:
                        ch_link_quality = 0.45
                    elif ch_snr >= 10:
                        ch_link_quality = 0.30
                    else:
                        ch_link_quality = 0.15
                    
                    # 能量状态影响
                    member_energy_status = node.remaining_energy / node.initial_energy
                    energy_factor = 0.3 + 0.7 * member_energy_status
                    
                    # 簇内通信成功率（更严格）
                    member_success_rate = (ch_link_quality * 0.6 + energy_factor * 0.4)
                   
                    
                    total_successful_packets += member_success_rate
    
        # 返回PDR（成功接收的数据包数 / 发送的数据包总数）
        pdr = total_successful_packets / round_packets_sent
        return  pdr  
    def perform_data_transmission(self):
        k = 4000
        round_packets_sent = 0
        round_packets_success = 0
    
        # 簇内通信
        for node in self.alive_nodes:
            if not node.is_cluster_head and node.cluster_id != -1:
                ch = next((n for n in self.alive_nodes if n.id == node.cluster_id and n.is_cluster_head), None)
                if ch and node.is_alive() and ch.is_alive():
                    round_packets_sent += 1

                    # 更严格的链路质量计算
                    distance = node.get_distance_to(ch.x, ch.y)
                    # 增加路径损耗，使成功率更现实
                    path_loss = 25 * math.log10(distance + 1) + 10 * math.log10(distance + 1)
                    snr = max(5, 85 - path_loss)  # 降低基础SNR
                    
                    # 更保守的链路质量评估
                    if snr >= 25:
                        link_quality = 0.80
                    elif snr >= 20:
                        link_quality = 0.65
                    elif snr >= 15:
                        link_quality = 0.45
                    elif snr >= 10:
                        link_quality = 0.30
                    else:
                        link_quality = 0.15

                    # 能量状态影响更明显
                    energy_status = node.remaining_energy / node.initial_energy
                    energy_factor = 0.3 + 0.7 * energy_status
                    
                    # 恶意节点影响
                    malicious_factor = 0.4 if node.is_malicious else 1.0
                    
                    # 综合成功率（更严格）
                    success_rate = (link_quality * 0.6 + energy_factor * 0.4) * malicious_factor
                    
                    success = random.random() < success_rate
                    if success:
                        round_packets_success += 1
                        tx_energy = self.energy_model.calculate_transmit_energy(k, distance)
                        rx_energy = self.energy_model.calculate_receive_energy(k)
                        node.consume_energy(tx_energy)
                        ch.consume_energy(rx_energy)
                        node.packets_sent += 1
                        ch.packets_received += 1
                    else:
                        node.packets_dropped += 1
    
        # 簇头到基站通信
        for node in self.alive_nodes:
            if node.is_cluster_head and node.is_alive():
                round_packets_sent += 1
                distance_to_bs = node.get_distance_to(self.bs_x, self.bs_y)

                # 更严格的基站通信链路质量
                path_loss = 30 * math.log10(distance_to_bs + 1) + 15 * math.log10(distance_to_bs + 1)
                snr = max(5, 80 - path_loss)  # 进一步降低基站通信SNR
                
                # 更保守的基站通信成功率
                if snr >= 25:
                    link_quality = 0.75
                elif snr >= 20:
                    link_quality = 0.60
                elif snr >= 15:
                    link_quality = 0.40
                elif snr >= 10:
                    link_quality = 0.25
                else:
                    link_quality = 0.10

                # 能量状态影响
                energy_status = node.remaining_energy / node.initial_energy
                energy_factor = 0.2 + 0.8 * energy_status
                
                # 恶意节点影响更严重
                if node.is_malicious:
                    if random.random() < 0.6:  # 60%概率丢包
                        node.packets_dropped += 1
                        continue
                    malicious_factor = 0.3
                else:
                    malicious_factor = 1.0

                # 综合成功率
                success_rate = (link_quality * 0.5 + energy_factor * 0.5) * malicious_factor
                

                success = random.random() < success_rate
                if success:
                    round_packets_success += 1
                    aggr_energy = self.energy_model.calculate_aggregation_energy(k)
                    tx_energy = self.energy_model.calculate_transmit_energy(k, distance_to_bs)
                    node.consume_energy(aggr_energy + tx_energy)
                    node.packets_sent += 1
                else:
                    node.packets_dropped += 1
     
        # 计算 PDR
        if round_packets_sent > 0:
            pdr = round_packets_success / round_packets_sent
        else:
            pdr = 0.0  # 没有发送包时PDR为0
    
        self.pdr_data.append(pdr * 100)  # 存百分比
        self.packets_transferred += round_packets_success
        self.packets_dropped += (round_packets_sent - round_packets_success)

        
    def update_network_statistics(self):
        """更新网络统计信息"""
        self.alive_nodes = [node for node in self.nodes if node.is_alive()]
        self.dead_nodes = [node for node in self.nodes if not node.is_alive()]
        
        self.alive_nodes_data.append(len(self.alive_nodes))
        
        total_energy = sum(node.remaining_energy for node in self.alive_nodes)
        self.residual_energy_data.append(total_energy)
        
        if len(self.alive_nodes) > 1:
            energies = [node.remaining_energy for node in self.alive_nodes]
            energy_std = np.std(energies)
            self.energy_std_data.append(energy_std)
        else:
            self.energy_std_data.append(0)
        
        # 更新网络寿命指标
        if len(self.alive_nodes) < len(self.nodes) and self.network_lifetime_metrics['fnd'] == -1:
            self.network_lifetime_metrics['fnd'] = self.current_round
            
        if len(self.alive_nodes) < len(self.nodes) // 2 and self.network_lifetime_metrics['hnd'] == -1:
            self.network_lifetime_metrics['hnd'] = self.current_round
            
        if len(self.alive_nodes) == 0 and self.network_lifetime_metrics['lnd'] == -1:
            self.network_lifetime_metrics['lnd'] = self.current_round
            
    def run_simulation(self, max_rounds: int = 1000):
        """运行完整仿真"""
        for round_num in range(1, max_rounds + 1):
            if len(self.alive_nodes) == 0:
               remaining_rounds = max_rounds - round_num + 1
               for _ in range(remaining_rounds):
                   self.alive_nodes_data.append(0)
                   self.residual_energy_data.append(0)
                   self.pdr_data.append(0)
                   self.energy_std_data.append(0)
               break
            self.run_clustering_round()
        while len(self.alive_nodes_data) < max_rounds:
            self.alive_nodes_data.append(0)
        while len(self.residual_energy_data) < max_rounds:
            self.residual_energy_data.append(0)
        while len(self.pdr_data) < max_rounds:
            self.pdr_data.append(0)
        while len(self.energy_std_data) < max_rounds:
            self.energy_std_data.append(0)    
        # 返回与FD-MFAR.py中其他协议相同格式的数据
        return (
            self.alive_nodes_data,
            self.residual_energy_data,
            self.pdr_data,
            self.energy_std_data,
            self.network_lifetime_metrics['fnd'],
            self.network_lifetime_metrics['hnd'],
            self.network_lifetime_metrics['lnd'],
            self.packets_transferred,
            self.packets_dropped
        )

    def run_simulation_until_death(self, print_interval: int = 100, max_rounds: int = 1000):
        """运行仿真直到所有节点死亡或达到最大轮数"""
        print(f"开始BESCR协议寿命仿真 - {len(self.nodes)}节点网络")
        print("="*80)
        print("轮数    | 存活节点 | 当前PDR | 近100轮平均PDR | 传输包数 | 丢包数 | FND | HND | LND")
        print("-"*80)
        
        round_num = 1
        while len(self.alive_nodes) > 0 and round_num <= max_rounds:
            self.run_clustering_round()
            
            # 每print_interval轮打印一次PDR和网络寿命指标
            if round_num % print_interval == 0:
                # 计算最近print_interval轮的平均PDR
                recent_pdr = self.pdr_data[-print_interval:] if len(self.pdr_data) >= print_interval else self.pdr_data
                avg_pdr = np.mean(recent_pdr) if recent_pdr else 0
                
                # 计算当前轮次的PDR
                current_pdr = self.pdr_data[-1] if self.pdr_data else 0
                
                # 获取网络寿命指标
                fnd = self.network_lifetime_metrics['fnd'] if self.network_lifetime_metrics['fnd'] != -1 else "未发生"
                hnd = self.network_lifetime_metrics['hnd'] if self.network_lifetime_metrics['hnd'] != -1 else "未发生"
                lnd = self.network_lifetime_metrics['lnd'] if self.network_lifetime_metrics['lnd'] != -1 else "未发生"
                
                print(f"第 {round_num:4d} 轮 | {len(self.alive_nodes):6d} | {current_pdr:6.2f}% | {avg_pdr:12.2f}% | {self.packets_transferred:8d} | {self.packets_dropped:6d} | {fnd:>3} | {hnd:>3} | {lnd:>3}")
            
            round_num += 1
        
        # 打印最终的网络寿命指标
        print("\n" + "="*80)
        print("BESCR协议网络寿命指标总结")
        print("="*80)
        fnd = self.network_lifetime_metrics['fnd']
        hnd = self.network_lifetime_metrics['hnd']
        lnd = self.network_lifetime_metrics['lnd']
        
        if fnd != -1:
            print(f"第一个节点死亡轮数 (FND): 第 {fnd} 轮")
        else:
            print("第一个节点死亡轮数 (FND): 仿真期间未发生")
            
        if hnd != -1:
            print(f"半数节点死亡轮数 (HND): 第 {hnd} 轮")
        else:
            print("半数节点死亡轮数 (HND): 仿真期间未发生")
            
        if lnd != -1:
            print(f"全部节点死亡轮数 (LND): 第 {lnd} 轮")
        else:
            print("全部节点死亡轮数 (LND): 仿真期间未发生")
        
        print(f"最终存活节点数: {len(self.alive_nodes)}")
        print(f"总传输包数: {self.packets_transferred}")
        print(f"总丢包数: {self.packets_dropped}")
        print(f"总仿真轮数: {round_num - 1}")
        print("="*80)
        
        # 返回字典格式的数据，用于网络寿命实验
        return {
            'alive_nodes_history': self.alive_nodes_data,
            'residual_energy_history': self.residual_energy_data,
            'pdr_history': self.pdr_data,
            'energy_std_history': self.energy_std_data,
            'fnd': self.network_lifetime_metrics['fnd'],
            'hnd': self.network_lifetime_metrics['hnd'],
            'lnd': self.network_lifetime_metrics['lnd'],
            'total_packets_transferred': self.packets_transferred,
            'total_packets_dropped': self.packets_dropped,
            'total_rounds': round_num - 1,
            'final_pdr': self.pdr_data[-1] if self.pdr_data else 0,
            'print_interval': print_interval
        }

def run_bescr():
    """运行BESCR协议的包装函数，与FD-MFAR.py中的其他协议函数格式一致"""
    global n, rmax
    protocol = BESCRProtocol(network_size=n)
    protocol.initialize_network(n)
    return protocol.run_simulation(rmax)

def run_bescr_until_death():
    """运行BESCR协议直到所有节点死亡，用于网络寿命对比实验"""
    global n, rmax
    
    # 如果全局变量不存在，设置默认值
    if 'n' not in globals():
        n = 400
    if 'rmax' not in globals():
        rmax = 1000
        
    original_rmax = rmax
    rmax = 20000  # 设置一个很大的值确保运行到所有节点死亡
    
    try:
        protocol = BESCRProtocol(network_size=n)
        protocol.initialize_network(n)
        alive_data, _, _, _, _, _, _, _, _ = protocol.run_simulation(rmax)
        
        # 找到最后一个非零值的索引（即最后一个节点死亡的轮数）
        last_alive_index = len(alive_data) - 1
        while last_alive_index >= 0 and alive_data[last_alive_index] == 0:
            last_alive_index -= 1
        
        # 截取到最后一个节点死亡的轮数
        if last_alive_index >= 0:
            alive_data = alive_data[:last_alive_index + 1]
        
        print(f"BESCR协议完成，运行了 {len(alive_data)} 轮")
        return alive_data
        
    finally:
        rmax = original_rmax  # 恢复原始rmax值

def run_lifetime_comparison_experiment():
    """
    对比BESCR、FDMMBF-ARP、LEACH、HEED、D2CRP协议的网络整体寿命（FND/HND/LND）。
    参考FD-MFAR.py的实现，但只运行BESCR协议进行测试。
    """
    # 定义默认值，避免全局变量未定义的问题
    global n, is_display
    
    # 如果全局变量不存在，设置默认值
    if 'n' not in globals():
        n = 400
    if 'is_display' not in globals():
        is_display = False
        
    original_n = n
    original_is_display = is_display

    n = 400
    is_display = False

    original_font_family = plt.rcParams['font.family']
    original_font_size = plt.rcParams['font.size']
    
    # 为英文图表设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    print(f"\n--- 运行BESCR网络寿命实验 (节点数={n}) ---")
    print("注意：BESCR协议将运行到所有节点死亡为止，可能需要较长时间...")

    def calc_lifetime(alive_arr):
        """计算网络寿命指标"""
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

    # 运行BESCR协议
    print("正在运行 BESCR 协议...")
    alive_bescr = run_bescr_until_death()
    
    # 计算FND/HND/LND
    fnd_bescr, hnd_bescr, lnd_bescr = calc_lifetime(alive_bescr)

    # 打印详细结果
    print("\n=== BESCR网络寿命结果 ===")
    print(f"{'协议':<12} {'FND':<8} {'HND':<8} {'LND':<8} {'总轮数':<8}")
    print("-" * 50)
    print(f"{'BESCR':<12} {fnd_bescr:<8} {hnd_bescr:<8} {lnd_bescr:<8} {len(alive_bescr):<8}")

    # 绘制BESCR网络寿命曲线
    plt.figure(figsize=(12, 8))
    
    # 创建时间轴
    rounds = np.arange(len(alive_bescr))
    
    # 绘制存活节点数曲线
    plt.plot(rounds, alive_bescr, 'D-', label='BESCR', linewidth=2, markersize=4, color='#E74C3C')
    
    # 标记关键寿命指标
    if fnd_bescr < len(alive_bescr):
        plt.axvline(x=fnd_bescr, color='red', linestyle='--', alpha=0.7, label=f'FND (第{fnd_bescr}轮)')
    if hnd_bescr < len(alive_bescr):
        plt.axvline(x=hnd_bescr, color='orange', linestyle='--', alpha=0.7, label=f'HND (第{hnd_bescr}轮)')
    if lnd_bescr < len(alive_bescr):
        plt.axvline(x=lnd_bescr, color='purple', linestyle='--', alpha=0.7, label=f'LND (第{lnd_bescr}轮)')
    
    plt.xlabel('rounds', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Alive Nodes', fontsize=14, fontweight='bold')
    plt.title('BESCR Protocol - Network Lifetime Analysis', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('bescr_lifetime_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制网络寿命指标柱状图
    plt.figure(figsize=(10, 6))
    
    metrics = ['FND', 'HND', 'LND']
    values = [fnd_bescr, hnd_bescr, lnd_bescr]
    colors = ['#60B8B5', '#78B7C9', '#FF8C00']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 在柱顶添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('rounds', fontsize=12, fontfamily='Times New Roman')
    plt.title('BESCR Protocol - Network Lifetime Metrics', fontsize=14, fontfamily='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('bescr_lifetime_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 恢复全局变量
    n = original_n
    is_display = original_is_display
    plt.rcParams['font.family'] = original_font_family
    plt.rcParams['font.size'] = original_font_size
    print("\nBESCR网络寿命实验完成，全局变量已恢复。")


def run_bescr_experiment():
    """运行BESCR协议实验，测试不同节点数量下的数据包传输成功率"""
    # 设置实验参数
    node_counts = list(range(20, 101, 10))  # 节点数量从20到100，步长为10
    max_rounds = 3000
    area_size = 100.0
    
    # 存储实验结果
    results = {
        'node_counts': [],
        'pdr_values': [],
        'fnd_values': [],
        'hnd_values': [],
        'lnd_values': [],
        'total_packets_transferred': [],
        'total_packets_dropped': []
    }
    
    print("开始BESCR协议实验...")
    print("节点数量\tPDR(%)\t\tFND\t\tHND\t\tLND\t\t传输包数\t丢包数")
    print("-" * 80)
    
    for node_count in node_counts:
        print(f"正在测试 {node_count} 个节点...", end=" ")
        
        # 创建协议实例
        protocol = BESCRProtocol(network_size=node_count, area_size=area_size)
        protocol.initialize_network(node_count)
        
        # 运行仿真
        start_time = time.time()
        (alive_nodes_data, residual_energy_data, pdr_data, 
         energy_std_data, fnd, hnd, lnd, packets_transferred, packets_dropped) = protocol.run_simulation(max_rounds)
        end_time = time.time()
        
        # 计算平均PDR
        avg_pdr = np.mean(pdr_data) if pdr_data else 0
        
        # 存储结果
        results['node_counts'].append(node_count)
        results['pdr_values'].append(avg_pdr)
        results['fnd_values'].append(fnd)
        results['hnd_values'].append(hnd)
        results['lnd_values'].append(lnd)
        results['total_packets_transferred'].append(packets_transferred)
        results['total_packets_dropped'].append(packets_dropped)
        
        print(f"完成 (耗时: {end_time - start_time:.2f}秒)")
        print(f"{node_count}\t\t{avg_pdr:.2f}\t\t{fnd}\t\t{hnd}\t\t{lnd}\t\t{packets_transferred}\t\t{packets_dropped}")
    
    return results


def test_large_scale_networks():
    """测试大规模网络（350、400、450节点）的数据包传输成功率"""
    print("BESCR协议大规模网络测试")
    print("="*50)
    print("测试节点数: 350, 400, 450")
    print("网络区域: 200x200")
    print("最大轮数: 1000")
    print("="*50)
    
    # 测试节点数量
    node_counts = [350, 400, 450]
    max_rounds = 1000
    area_size = 200.0
    
    # 存储结果
    results = {
        'node_counts': [],
        'pdr_values': [],
        'fnd_values': [],
        'hnd_values': [],
        'lnd_values': [],
        'total_packets_transferred': [],
        'total_packets_dropped': [],
        'simulation_times': []
    }
    
    print("\n开始大规模网络测试...")
    print("节点数量\tPDR(%)\t\tFND\t\tHND\t\tLND\t\t传输包数\t丢包数\t\t耗时(秒)")
    print("-" * 90)
    
    for node_count in node_counts:
        print(f"正在测试 {node_count} 个节点...", end=" ")
        
        # 记录开始时间
        start_time = time.time()
        
        # 创建协议实例
        protocol = BESCRProtocol(network_size=node_count, area_size=area_size)
        protocol.initialize_network(node_count)
        
        # 运行仿真
        (alive_nodes_data, residual_energy_data, pdr_data, 
         energy_std_data, fnd, hnd, lnd, packets_transferred, packets_dropped) = protocol.run_simulation(max_rounds)
        
        # 记录结束时间
        end_time = time.time()
        simulation_time = end_time - start_time
        
        # 计算平均PDR
        avg_pdr = np.mean(pdr_data) if pdr_data else 0
        
        # 存储结果
        results['node_counts'].append(node_count)
        results['pdr_values'].append(avg_pdr)
        results['fnd_values'].append(fnd)
        results['hnd_values'].append(hnd)
        results['lnd_values'].append(lnd)
        results['total_packets_transferred'].append(packets_transferred)
        results['total_packets_dropped'].append(packets_dropped)
        results['simulation_times'].append(simulation_time)
        
        print(f"完成")
        print(f"{node_count}\t\t{avg_pdr:.2f}\t\t{fnd}\t\t{hnd}\t\t{lnd}\t\t{packets_transferred}\t\t{packets_dropped}\t\t{simulation_time:.2f}")
    
    # 绘制结果
    plot_large_scale_results(results)
    
    # 分析结果
    analyze_large_scale_performance(results)
    
    return results



def main():
    """主函数 - 支持多种实验模式"""
    print("BESCR协议性能测试实验")
    print("="*50)
    print("请选择实验模式:")
    print("1. 不同工作轮数数据包传输成功率测试 (20-100节点，步长10)")
    print("2. 不同节点数下数据包传输成功率测试 (350, 400, 450节点)")
    print("3. 网络整体寿命实验 (运行到所有节点死亡)")
    print("="*50)
    
    while True:
        try:
            choice = input("请输入选择 (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            else:
                print("请输入有效选择 (1-3)")
        except KeyboardInterrupt:
            print("\n程序已退出")
            return
    
    if choice == '1':
        print("\n运行不同工作轮数数据包传输成功率测试...")
        print("实验参数:")
        print("- 节点数量范围: 20-100 (步长: 10)")
        print("- 最大轮数: 3000")
        print("- 网络区域: 100x100")
        print("- 初始能量: 0.6J")
        print("- 恶意节点比例: 5%")
        print("="*50)
        
        results = run_bescr_experiment()
        plot_bescr_results(results)
        analyze_bescr_performance(results)
        
    elif choice == '2':
        print("\n不同节点数下数据包传输成功率测试...")
        print("实验参数:")
        print("- 节点数量: 350, 400, 450")
        print("- 最大轮数: 1000")
        print("- 网络区域: 200x200")
        print("- 初始能量: 0.6J")
        print("- 恶意节点比例: 5%")
        print("="*50)
        
        results = test_large_scale_networks()

    elif choice == '3':
        print("\n网络整体寿命实验...")
        print("实验参数:")
        print("- 节点数量: 400")
        print("- 最大轮数: 20000")
        print("- 网络区域: 100x100")
        print("- 初始能量: 0.6J")
        print("- 恶意节点比例: 5%")
        print("="*50)
        
        results = run_lifetime_comparison_experiment()
        

if __name__ == "__main__":
   
   n=400
   rmax = 20000
   is_display = False
   RUN_LIFETIME_COMPARISON_EXPERIMENT = True
   RUN_TEST_LARGE_SCALE_NETWORKS = False
   RUN_BESCR_EXPERIMENT = False

   if RUN_LIFETIME_COMPARISON_EXPERIMENT:
        results = run_lifetime_comparison_experiment()
   if RUN_TEST_LARGE_SCALE_NETWORKS:
        results = test_large_scale_networks()
   if RUN_BESCR_EXPERIMENT:
        results = run_bescr_experiment()
    

        


