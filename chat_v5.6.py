import torch
import torch.nn as nn
import json
import os
import random
import time
import threading
from collections import defaultdict

# ==================== 配置 ====================
VECTOR_DIM = 4096
LEARNING_RATE = 0.1
AUTO_SAVE_INTERVAL = 60
SPEED_CONTROL_ENABLED = True

# ==================== GPU 设备 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[设备] 使用: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")

# ==================== 维度语义定义 ====================
DIMENSION_ZONES = {
    'predefined': (0, 32),
    'knowledge': (32, 64),
    'learning': (64, 3564),
    'meta_learning': (3564, 4080),
    'buffer': (4080, 4096),
}

SEMANTIC_LABELS = {
    # 基础语义维度 (0-15)
    0: '词频',
    1: '情感正向',
    2: '情感负向',
    3: '情感强度',
    4: '需求类',
    5: '动作类',
    6: '状态类',
    7: '对象类',
    8: '时间相关',
    9: '空间相关',
    10: '程度修饰',
    11: '否定/疑问',
    12: '人称相关',
    13: '数量相关',
    14: '因果关联',
    15: '并列关联',
    
    # 扩展语义维度 (16-31)
    16: '逻辑推理',
    17: '条件判断',
    18: '假设推演',
    19: '对比关系',
    20: '递进关系',
    21: '转折关系',
    22: '因果关系',
    23: '目的意图',
    24: '方式方法',
    25: '范围限定',
    26: '顺序序列',
    27: '重要性',
    28: '确定性',
    29: '紧急性',
    30: '抽象性',
    31: '具体性',
    
    # 常识知识维度 (32-127)
    32: '常识-生理需求',
    33: '常识-安全需求',
    34: '常识-社交需求',
    35: '常识-尊重需求',
    36: '常识-自我实现',
    37: '常识-食物关联',
    38: '常识-睡眠关联',
    39: '常识-健康关联',
    40: '常识-情感关联',
    41: '常识-行为关联',
    # 42-127: 自动学习，无需预定义标签
}


# ==================== 神经网络定义 ====================
class WeightNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, vector_dim=4096):
        super().__init__()
        self.vector_dim = vector_dim
        
        self.stat_embedding = nn.Embedding(vocab_size, vector_dim)
        self.tuned_embedding = nn.Embedding(vocab_size, vector_dim)
        
        self.dimension_semantic_net = nn.Sequential(
            nn.Linear(vector_dim, vector_dim),
            nn.ReLU(),
            nn.Linear(vector_dim, vector_dim),
        )
        
        self.weight_relation = nn.Linear(vector_dim, vector_dim)
        
        self.combination_net = nn.Sequential(
            nn.Linear(vector_dim * 2, vector_dim),
            nn.ReLU(),
            nn.Linear(vector_dim, vector_dim)
        )
        
        self.relation_score_net = nn.Sequential(
            nn.Linear(vector_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.buffer_compute_net = nn.Sequential(
            nn.Linear(16,32 ),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        
        self.dimension_activations = defaultdict(list)
        self.learned_semantics = {}
        
    def get_weights(self, word_idx):
        stat_weights = self.stat_embedding(word_idx)
        tuned_weights = self.tuned_embedding(word_idx)
        return stat_weights, tuned_weights
    
    def get_combined_weight(self, word_idx):
        stat_w, tuned_w = self.get_weights(word_idx)
        return stat_w + tuned_w
    
    def compute_weight_relations(self, weights):
        return self.weight_relation(weights)
    
    def compute_combination_weight(self, w1_idx, w2_idx):
        w1 = self.get_combined_weight(w1_idx)
        w2 = self.get_combined_weight(w2_idx)
        combined = torch.cat([w1, w2], dim=-1)
        return self.combination_net(combined)
    
    def compute_relation_score(self, weights1, weights2):
        if weights1.dim() == 1:
            weights1 = weights1.unsqueeze(0)
        if weights2.dim() == 1:
            weights2 = weights2.unsqueeze(0)
        combined = torch.cat([weights1, weights2], dim=-1)
        return self.relation_score_net(combined)
    
    def compute_with_buffer(self, weights):
        buffer_zone = DIMENSION_ZONES['buffer']
        buffer_weights = weights[:, buffer_zone[0]:buffer_zone[1]]
        buffer_result = self.buffer_compute_net(buffer_weights)
        # 992维前面部分需要padding
        buffer_size = DIMENSION_ZONES['buffer'][1] - DIMENSION_ZONES['buffer'][0]
        pad_size = self.vector_dim - buffer_size
        return weights + torch.nn.functional.pad(buffer_result, (0, pad_size))
    
    def track_dimension_activation(self, word_idx, word=None):
        weights = self.get_combined_weight(word_idx)
        learning_zone = DIMENSION_ZONES['learning']
        learning_weights = weights[0, learning_zone[0]:learning_zone[1]]
        top_dims = torch.topk(learning_weights.abs(), k=3)
        
        if word:
            for dim_idx in top_dims.indices:
                actual_dim = learning_zone[0] + dim_idx.item()
                self.dimension_activations[actual_dim].append(word)
                
    def infer_dimension_semantics(self):
        print("\n[维度语义推断] 开始分析...")
        
        for dim, words in self.dimension_activations.items():
            if len(words) >= 5:
                word_freq = defaultdict(int)
                for w in words[-50:]:
                    word_freq[w] += 1
                    
                top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:5]
                if top_words:
                    self.learned_semantics[dim] = {
                        'words': [w for w, c in top_words],
                        'count': len(words),
                        'inferred_meaning': f"关联词: {', '.join([w for w, c in top_words[:3]])}"
                    }
                    
        learned_count = len(self.learned_semantics)
        print(f"[维度语义推断] 学习到 {learned_count} 个维度语义")
        
        for dim, info in list(self.learned_semantics.items())[:5]:
            print(f"  维度{dim}: {info['inferred_meaning']}")
            
        return self.learned_semantics


# ==================== 速度控制器 ====================
class SpeedController:
    def __init__(self, enabled=True):
        self.enabled = enabled and (DEVICE.type == 'cpu')
        self.batch_delay = 0.001
        self.epoch_delay = 0.01
        
    def control(self, stage='batch'):
        if not self.enabled:
            return
        if stage == 'batch':
            time.sleep(self.batch_delay)
        elif stage == 'epoch':
            time.sleep(self.epoch_delay)
            
    def adjust_speed(self, speed_level):
        if speed_level == 1:
            self.batch_delay = 0.01
            self.epoch_delay = 0.1
        elif speed_level == 5:
            self.batch_delay = 0.001
            self.epoch_delay = 0.01
        elif speed_level == 10:
            self.batch_delay = 0.0001
            self.epoch_delay = 0.001


# ==================== 进度显示器 ====================
class ProgressDisplay:
    def __init__(self, total_epochs, stage_name="训练"):
        self.total_epochs = total_epochs
        self.stage_name = stage_name
        self.start_time = time.time()
        self.epoch_times = []
        
    def update(self, epoch, loss, batch_info=""):
        elapsed = time.time() - self.start_time
        
        if len(self.epoch_times) > 0:
            avg_epoch_time = sum(self.epoch_times[-10:]) / min(len(self.epoch_times), 10)
            remaining_epochs = self.total_epochs - epoch - 1
            eta = avg_epoch_time * remaining_epochs
        else:
            eta = 0
            
        progress = (epoch + 1) / self.total_epochs * 100
        bar_len = 20
        filled = int(bar_len * (epoch + 1) / self.total_epochs)
        bar = '█' * filled + '░' * (bar_len - filled)
        
        eta_str = self._format_time(eta)
        elapsed_str = self._format_time(elapsed)
        
        if batch_info:
            print(f"\r[{self.stage_name}] {bar} {epoch+1}/{self.total_epochs} ({progress:.0f}%) | Loss: {loss:.4f} | 用时: {elapsed_str} | 预估剩余: {eta_str} | {batch_info}", end='', flush=True)
        else:
            print(f"\r[{self.stage_name}] {bar} {epoch+1}/{self.total_epochs} ({progress:.0f}%) | Loss: {loss:.4f} | 用时: {elapsed_str} | 预估剩余: {eta_str}", end='', flush=True)
        
    def epoch_done(self, epoch_time):
        self.epoch_times.append(epoch_time)
        
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\n[{self.stage_name}完成] 总用时: {self._format_time(elapsed)}")
        
    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


# ==================== 权重系统核心 ====================
class WeightSystem:
    def __init__(self, vector_dim=4096):
        self.vector_dim = vector_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
        self.word_freq = defaultdict(int)
        self.bigram_freq = defaultdict(lambda: defaultdict(int))
        self.sentence_patterns = []
        
        self.nn = None
        self.speed_controller = SpeedController(SPEED_CONTROL_ENABLED)
        
        self.combination_relations = defaultdict(lambda: defaultdict(float))
        self.group_relations = defaultdict(lambda: defaultdict(float))
        self.group_word_relations = defaultdict(lambda: defaultdict(float))
        self.big_groups = defaultdict(list)
        self.common_knowledge = {}
        self.dictionary = {}
        
        self.training_pairs = []
        self.auto_learned_relations = {}
        
    def build_vocab(self, texts):
        for text in texts:
            words = list(text)
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = self.vocab_size
                    self.idx_to_word[self.vocab_size] = word
                    self.vocab_size += 1
        self.nn = WeightNeuralNetwork(self.vocab_size, self.vector_dim).to(DEVICE)
        
    def collect_statistics(self, texts):
        for text in texts:
            words = list(text)
            for word in words:
                self.word_freq[word] += 1
            for i in range(len(words) - 1):
                self.bigram_freq[words[i]][words[i+1]] += 1
            self.sentence_patterns.append(words)
            
    def compute_statistical_weights(self):
        total_words = sum(self.word_freq.values())
        
        with torch.no_grad():
            for word, idx in self.word_to_idx.items():
                freq_ratio = self.word_freq[word] / total_words if total_words > 0 else 0
                
                stat_weights = torch.zeros(self.vector_dim)
                stat_weights[0] = freq_ratio
                
                if word in self.dictionary:
                    dict_info = self.dictionary[word]
                    emotion = dict_info.get('emotion', 0)
                    category = dict_info.get('category', '')
                    
                    if emotion > 0:
                        stat_weights[1] = emotion
                    elif emotion < 0:
                        stat_weights[2] = -emotion
                    stat_weights[3] = abs(emotion)
                    
                    category_map = {
                        '需求': 4, '动作': 5, '状态': 6, '对象': 7,
                        '时间': 8, '空间': 9, '程度': 10, '疑问': 11,
                        '人称': 12, '数量': 13,
                    }
                    if category in category_map:
                        stat_weights[category_map[category]] = 0.8
                        
                # 4096维分区初始化
                stat_weights[32:64] = torch.randn(32) * 0.05       # knowledge区
                stat_weights[64:3564] = torch.zeros(3500)          # learning区
                stat_weights[3564:4080] = torch.zeros(516)          # meta_learning区
                stat_weights[4080:4096] = torch.zeros(16)          # buffer区
                
                self.nn.stat_embedding.weight[idx] = stat_weights
                
    def get_word_weight(self, word):
        if word not in self.word_to_idx:
            return None
        idx = torch.tensor([self.word_to_idx[word]], device=DEVICE)
        return self.nn.get_combined_weight(idx)
    
    def get_weight_relations(self, word):
        weights = self.get_word_weight(word)
        if weights is None:
            return None
        return self.nn.compute_weight_relations(weights)
    
    def compute_combination_ratio(self, w1, w2):
        if w1 in self.bigram_freq:
            total = sum(self.bigram_freq[w1].values())
            stat_ratio = self.bigram_freq[w1][w2] / total if total > 0 else 0
        else:
            stat_ratio = 0
            
        if w1 in self.word_to_idx and w2 in self.word_to_idx:
            idx1 = torch.tensor([self.word_to_idx[w1]], device=DEVICE)
            idx2 = torch.tensor([self.word_to_idx[w2]], device=DEVICE)
            nn_weight = self.nn.compute_combination_weight(idx1, idx2)
        else:
            nn_weight = torch.zeros(self.vector_dim, device=DEVICE)
            
        return stat_ratio, nn_weight
    
    def compute_combination_relations(self, combo1, combo2):
        cooccur = 0
        combo1_count = 0
        
        for pattern in self.sentence_patterns:
            pattern_str = ''.join(pattern)
            has1 = combo1 in pattern_str
            has2 = combo2 in pattern_str
            if has1:
                combo1_count += 1
            if has1 and has2:
                cooccur += 1
                
        relation = cooccur / combo1_count if combo1_count > 0 else 0
        self.combination_relations[combo1][combo2] = relation
        return relation
    
    def compute_group_relations(self, big_group_name, small_group):
        if big_group_name not in self.big_groups:
            return 0.0
            
        big_content = ''.join(self.big_groups[big_group_name])
        small_content = small_group
        
        big_weights = []
        small_weights = []
        
        for word in big_content:
            if word in self.word_to_idx:
                idx = torch.tensor([self.word_to_idx[word]], device=DEVICE)
                big_weights.append(self.nn.get_combined_weight(idx))
                
        for word in small_content:
            if word in self.word_to_idx:
                idx = torch.tensor([self.word_to_idx[word]], device=DEVICE)
                small_weights.append(self.nn.get_combined_weight(idx))
                
        if big_weights and small_weights:
            big_avg = torch.mean(torch.stack([w.squeeze(0) for w in big_weights]), dim=0)
            small_avg = torch.mean(torch.stack([w.squeeze(0) for w in small_weights]), dim=0)
            relation = self.nn.compute_relation_score(big_avg, small_avg).item()
            self.group_relations[big_group_name][small_group] = relation
            return relation
            
        return 0.0
    
    def compute_group_word_relations(self, big_group_name, word):
        if big_group_name not in self.big_groups:
            return 0.0
            
        combos = self.big_groups[big_group_name]
        word_count = 0
        total_chars = 0
        
        for combo in combos:
            word_count += combo.count(word)
            total_chars += len(combo)
            
        base_relation = word_count / total_chars if total_chars > 0 else 0
        
        if word in self.word_to_idx:
            idx = torch.tensor([self.word_to_idx[word]], device=DEVICE)
            word_weight = self.nn.get_combined_weight(idx)
            
            group_weights = []
            for combo in combos:
                for c_word in combo:
                    if c_word in self.word_to_idx:
                        c_idx = torch.tensor([self.word_to_idx[c_word]], device=DEVICE)
                        group_weights.append(self.nn.get_combined_weight(c_idx))
                        
            if group_weights:
                group_avg = torch.mean(torch.stack([w.squeeze(0) for w in group_weights]), dim=0)
                nn_relation = self.nn.compute_relation_score(
                    word_weight.squeeze(0), group_avg
                ).item()
                final_relation = base_relation * 0.5 + nn_relation * 0.5
            else:
                final_relation = base_relation
        else:
            final_relation = base_relation
            
        self.group_word_relations[big_group_name][word] = final_relation
        return final_relation
    
    def set_common_knowledge(self, word, related_word, weight=0.95):
        if word not in self.common_knowledge:
            self.common_knowledge[word] = {}
        self.common_knowledge[word][related_word] = weight
        
    def auto_learn_relation(self, w1, w2, context_score):
        key = f"{w1}→{w2}"
        
        if key in self.auto_learned_relations:
            old_score, count = self.auto_learned_relations[key]
            new_score = (old_score * count + context_score) / (count + 1)
            self.auto_learned_relations[key] = (new_score, count + 1)
        else:
            self.auto_learned_relations[key] = (context_score, 1)
            
        score, count = self.auto_learned_relations[key]
        if score > 0.7 and count >= 5:
            self.set_common_knowledge(w1, w2, score)
            print(f"\n[自动学习] 发现新关联: {w1}→{w2} = {score:.3f}")
            
    def form_big_group(self, group_name, combinations):
        self.big_groups[group_name] = combinations
        for combo in combinations:
            self.compute_group_relations(group_name, combo)
            for word in combo:
                self.compute_group_word_relations(group_name, word)
            
    def apply_common_knowledge_training(self, epochs=5):
        print(f"\n[常识训练] 开始 {epochs} 轮...")
        
        if not self.common_knowledge:
            print("[常识训练] 无数据")
            return
            
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.01)
        progress = ProgressDisplay(epochs, "常识训练")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            count = 0
            
            for word, relations in self.common_knowledge.items():
                if word not in self.word_to_idx:
                    continue
                    
                for related_word, target_relation in relations.items():
                    if related_word not in self.word_to_idx:
                        continue
                    
                    optimizer.zero_grad()
                    
                    word_idx = torch.tensor([self.word_to_idx[word]], device=DEVICE)
                    related_idx = torch.tensor([self.word_to_idx[related_word]], device=DEVICE)
                    
                    word_weight = self.nn.get_combined_weight(word_idx)
                    related_weight = self.nn.get_combined_weight(related_idx)
                    
                    current = self.nn.compute_relation_score(word_weight, related_weight)
                    
                    target = torch.tensor([[target_relation]], device=DEVICE)
                    loss = torch.nn.functional.mse_loss(current, target)
                    
                    loss.backward()
                    optimizer.step()
                    
                    self.nn.track_dimension_activation(word_idx, word)
                    
                    total_loss += loss.item()
                    count += 1
                    
                self.speed_controller.control('batch')
            
            avg_loss = total_loss / count if count > 0 else 0
            epoch_time = time.time() - epoch_start
            progress.update(epoch, avg_loss)
            progress.epoch_done(epoch_time)
            self.speed_controller.control('epoch')
                
        progress.finish()


# ==================== 循环迭代验证 ====================
class IterativeValidator:
    def __init__(self, weight_system, threshold=0.5, max_iterations=10):
        self.ws = weight_system
        self.threshold = threshold
        self.max_iterations = max_iterations
        
    def validate_and_iterate(self, input_words, predictions):
        best_response = None
        best_score = 0
        iteration = 0
        current_predictions = predictions.copy()
        
        input_weights = []
        for word in input_words:
            w = self.ws.get_word_weight(word)
            if w is not None:
                input_weights.append(w)
                
        if not input_weights:
            return predictions[0] if predictions else "..."
            
        input_avg = torch.mean(torch.stack([w.squeeze(0) for w in input_weights]), dim=0)
        
        while iteration < self.max_iterations:
            iteration += 1
            
            for response in current_predictions:
                response_words = list(response)
                response_weights = []
                
                for word in response_words:
                    w = self.ws.get_word_weight(word)
                    if w is not None:
                        response_weights.append(w)
                        
                if response_weights:
                    response_avg = torch.mean(torch.stack([w.squeeze(0) for w in response_weights]), dim=0)
                    
                    score = self.ws.nn.compute_relation_score(input_avg, response_avg).item()
                    
                    for iw in input_weights:
                        for rw in response_words:
                            if iw in self.ws.word_to_idx and rw in self.ws.word_to_idx:
                                self.ws.auto_learn_relation(iw, rw, score)
                    
                    if score > best_score:
                        best_score = score
                        best_response = response
                        
                    if score >= self.threshold:
                        print(f"\n[迭代{iteration}] 关联度{score:.4f} >= 阈值{self.threshold}, 采用")
                        return response
                        
            new_preds = self._regenerate(input_words)
            if new_preds:
                current_predictions = new_preds
            else:
                break
                
        print(f"\n[迭代结束] 最高关联度: {best_score:.4f}")
        return best_response if best_response else (predictions[0] if predictions else "...")
        
    def _regenerate(self, input_words):
        new_preds = []
        
        for word in input_words:
            if word not in self.ws.word_to_idx:
                continue
                
            word_idx = torch.tensor([self.ws.word_to_idx[word]], device=DEVICE)
            word_weight = self.ws.nn.get_combined_weight(word_idx)
            
            for other_word, other_idx in self.ws.word_to_idx.items():
                if other_word == word:
                    continue
                    
                other_idx_tensor = torch.tensor([other_idx], device=DEVICE)
                other_weight = self.ws.nn.get_combined_weight(other_idx_tensor)
                
                relation = self.ws.nn.compute_relation_score(
                    word_weight.squeeze(0), other_weight.squeeze(0)
                ).item()
                
                if relation > 0.3:
                    new_preds.append(f"你是说{word}{other_word}吗？")
                    
        return new_preds[:3]


# ==================== 自动保存管理器 ====================
class AutoSaveManager:
    def __init__(self, chat_system, interval=AUTO_SAVE_INTERVAL):
        self.chat_system = chat_system
        self.interval = interval
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.thread.start()
        print(f"[自动保存] 已启动，间隔 {self.interval} 秒")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        print("[自动保存] 已停止")
        
    def _auto_save_loop(self):
        while self.running:
            time.sleep(self.interval)
            if self.running:
                try:
                    self.chat_system.save_data(silent=True)
                    print(f"\n[自动保存] {time.strftime('%H:%M:%S')} 完成")
                except Exception as e:
                    print(f"\n[自动保存] 错误: {e}")


# ==================== 完整对话系统 ====================
class ChatSystem:
    # ==================== 架构测试模块 ====================
    def test_weight_relation(self):
        """测试权重关联强度"""
        print("\n" + "=" * 60)
        print("[测试] 权重关联强度")
        print("=" * 60)
    
        pairs = [
            ("饿", "吃"),    # 应该高
            ("饿", "飞"),    # 应该低
            ("困", "睡"),    # 应该高
            ("困", "跑"),    # 应该低
            ("开心", "笑"),  # 应该高
            ("开心", "哭"),  # 应该低
            ("难过", "哭"),  # 应该高
            ("难过", "笑"),  # 应该低
        ]
    
        results = []
        for w1, w2 in pairs:
            if w1 in self.weight_system.word_to_idx and w2 in self.weight_system.word_to_idx:
                w1_weight = self.weight_system.get_word_weight(w1)
                w2_weight = self.weight_system.get_word_weight(w2)
            
                if w1_weight is not None and w2_weight is not None:
                    score = self.weight_system.nn.compute_relation_score(
                        w1_weight.squeeze(0), w2_weight.squeeze(0)
                    ).item()
                    results.append((w1, w2, score))
                    status = "✅" if score > 0.5 else "⚠️"
                    print(f"  {w1} → {w2}: {score:.4f} {status}")
            else:
                print(f"  {w1} 或 {w2} 不在词表中")
    
        # 统计
        high_count = sum(1 for _, _, s in results if s > 0.5)
        print(f"\n[统计] 关联分>0.5: {high_count}/{len(results)}")
    
        return results

    def test_dimension_semantic(self):
        """测试维度语义映射"""
        print("\n" + "=" * 60)
        print("[测试] 维度语义映射")
        print("=" * 60)
    
        words = ["开心", "难过", "饿", "困", "累", "生气", "你好"]
    
        print(f"\n{'词语':<6} {'情感正向':<10} {'情感负向':<10} {'需求类':<10} {'状态类':<10}")
        print("-" * 50)
    
        for word in words:
            weight = self.weight_system.get_word_weight(word)
            if weight is not None:
                dim1 = weight[0, 1].item()  # 情感正向
                dim2 = weight[0, 2].item()  # 情感负向
                dim4 = weight[0, 4].item()  # 需求类
                dim6 = weight[0, 6].item()  # 状态类
                print(f"{word:<6} {dim1:<10.4f} {dim2:<10.4f} {dim4:<10.4f} {dim6:<10.4f}")
            else:
                print(f"{word:<6} 不在词表中")
    
        return True

    def test_learning_zone(self):
        """测试自主学习区"""
        print("\n" + "=" * 60)
        print("[测试] 自主学习区 (64-3564)")
        print("=" * 60)
    
        learned = self.weight_system.nn.learned_semantics
        learning_zone = DIMENSION_ZONES['learning']
    
        # 统计学习区内的维度
        learning_learned = {dim: info for dim, info in learned.items() 
                if learning_zone[0] <= dim < learning_zone[1]}
    
        print(f"\n[统计] 学习区已学习维度: {len(learning_learned)}/864")
        print(f"[利用率] {len(learning_learned)/3500*100:.1f}%")
    
        # 显示部分维度
        print("\n[示例] 前10个已学习维度:")
        for dim, info in sorted(learning_learned.items())[:10]:
            print(f"  维度{dim}: {info['inferred_meaning']}")
    
        # 检查相似词是否聚集
        print("\n[分析] 相似词聚集检测:")
        word_dims = {}
        for dim, info in learning_learned.items():
            for word in info['words'][:2]:
                if word not in word_dims:
                    word_dims[word] = []
                word_dims[word].append(dim)
    
        # 找重复出现的词
        repeated = {w: dims for w, dims in word_dims.items() if len(dims) > 1}
        if repeated:
            print(f"  多维度激活词: {list(repeated.keys())[:5]}")
        else:
            print("  暂无多维度激活词")
    
        return learning_learned

    def test_reasoning_chain(self):
        """测试推理链路"""
        print("\n" + "=" * 60)
        print("[测试] 推理链路")
        print("=" * 60)
    
        # 因果相关词对
        chains = [
            ("饿", "吃", "因果"),
            ("困", "睡", "因果"),
            ("累", "休息", "因果"),
            ("开心", "笑", "关联"),
            ("难过", "哭", "关联"),
        ]
    
        print(f"\n{'词1':<6} {'词2':<6} {'类型':<6} {'因果维度(14)':<12} {'因果维度(22)':<12} {'关联分':<10}")
        print("-" * 60)
    
        for w1, w2, chain_type in chains:
            w1_weight = self.weight_system.get_word_weight(w1)
            w2_weight = self.weight_system.get_word_weight(w2)
        
            if w1_weight is not None and w2_weight is not None:
                dim14 = w1_weight[0, 14].item()  # 因果关联
                dim22 = w1_weight[0, 22].item()  # 因果关系
                score = self.weight_system.nn.compute_relation_score(
                    w1_weight.squeeze(0), w2_weight.squeeze(0)
                ).item()
                print(f"{w1:<6} {w2:<6} {chain_type:<6} {dim14:<12.4f} {dim22:<12.4f} {score:<10.4f}")
            else:
                print(f"{w1:<6} {w2:<6} {chain_type:<6} 词不在词表中")
    
        return True

    def run_all_tests(self):
        """运行所有架构测试"""
        print("\n" + "=" * 60)
        print("架构核心测试")
        print("=" * 60)
    
        self.test_weight_relation()
        self.test_dimension_semantic()
        self.test_learning_zone()
        self.test_reasoning_chain()
    
        print("\n" + "=" * 60)
        print("[测试完成]")
        print("=" * 60)
    def __init__(self, vector_dim=4096, model_path="chat_v5.0_model.pt", data_path="chat_v5.0_data.json"):
        self.vector_dim = vector_dim
        self.model_path = model_path
        self.data_path = data_path
        
        self.weight_system = WeightSystem(vector_dim)
        self.validator = None
        self.auto_saver = None
        
        self.response_templates = {
            "饿": ["你饿了？饿多长时间了？", "想吃点什么？", "要不要吃点东西？"],
            "困": ["困了？昨晚没睡好吗？", "要不要休息一下？", "想睡觉了吗？"],
            "难过": ["怎么了？发生什么事了？", "想聊聊吗？", "我在这里陪你。"],
            "开心": ["有什么开心的事？", "太好了！分享一下吧？", "开心最重要！"],
            "高兴": ["有什么高兴的事？", "太棒了！", "恭喜恭喜！"],
            "伤心": ["怎么了？谁欺负你了？", "想哭就哭出来吧。", "我在这里。"],
            "生气": ["谁惹你生气了？", "消消气，慢慢说。", "发生什么事了？"],
            "害怕": ["别怕，有我在。", "怎么了？发生什么了？", "我在你身边。"],
            "累": ["辛苦了，休息一下吧。", "要不要放松一下？", "今天很忙吗？"],
            "痛": ["哪里痛？要不要看医生？", "很痛吗？", "需要帮忙吗？"],
            "你好": ["你好！有什么想聊的吗？", "嗨！今天怎么样？", "你好呀！"],
            "谢谢": ["不客气！", "应该的！", "有需要随时找我。"],
            "再见": ["再见！下次聊！", "拜拜！", "期待下次见面！"],
            "喜欢": ["喜欢什么？", "真好！", "我也喜欢！"],
            "爱": ["我也爱你！", "谢谢你！", "真感动！"],
            "孤独": ["想聊聊吗？", "我在这里。", "要不要找朋友？"],
            "焦虑": ["放松一下。", "有什么担心的？", "深呼吸。"],
            "迷茫": ["怎么了？", "有什么困惑？", "我们一起想办法。"],
        }
        
        self.correction_count = 0
        self.dialogue_history = []
        
        self.load_data()
        
    def load_data(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.weight_system.word_freq = defaultdict(int, data.get('word_freq', {}))
                self.weight_system.bigram_freq = defaultdict(
                    lambda: defaultdict(int),
                    {k: defaultdict(int, v) for k, v in data.get('bigram_freq', {}).items()}
                )
                self.weight_system.sentence_patterns = data.get('sentence_patterns', [])
                self.correction_count = data.get('correction_count', 0)
                self.dialogue_history = data.get('dialogue_history', [])
                self.weight_system.big_groups = defaultdict(list, data.get('big_groups', {}))
                self.weight_system.common_knowledge = data.get('common_knowledge', {})
                self.weight_system.dictionary = data.get('dictionary', {})
                self.weight_system.training_pairs = data.get('training_pairs', [])
                self.weight_system.auto_learned_relations = data.get('auto_learned_relations', {})
                    
                print(f"[加载] 词频: {len(self.weight_system.word_freq)} 词")
                print(f"[加载] 自动学习关联: {len(self.weight_system.auto_learned_relations)} 条")
                
        if os.path.exists(self.model_path):
            if self.weight_system.word_freq:
                self.weight_system.build_vocab(list(self.weight_system.word_freq.keys()))
            self.weight_system.compute_statistical_weights()
            
            try:
                checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
                model_state = self.weight_system.nn.state_dict()
                filtered_state = {k: v for k, v in checkpoint['model_state'].items() 
                                  if k in model_state and v.shape == model_state[k].shape}
                self.weight_system.nn.load_state_dict(filtered_state, strict=False)
                
                if 'dimension_activations' in checkpoint:
                    for dim, words in checkpoint['dimension_activations'].items():
                        self.weight_system.nn.dimension_activations[int(dim)] = words
                        
                if 'learned_semantics' in checkpoint:
                    self.weight_system.nn.learned_semantics = {int(k): v for k, v in checkpoint['learned_semantics'].items()}
                    
                print("[加载] 神经网络参数已恢复")
            except Exception as e:
                print(f"[加载] 神经网络参数不兼容，使用新参数")
        else:
            print("[初始化] 新系统")
            
        self.validator = IterativeValidator(self.weight_system)
        
    def save_data(self, silent=False):
        # 先推理最新语义，确保数据是最新的
        if not silent:
            print("[保存前] 推理维度语义...")
            self.weight_system.nn.infer_dimension_semantics()
        bigram_serializable = {k: dict(v) for k, v in self.weight_system.bigram_freq.items()}
        
        dim_activations = {}
        for dim, words in self.weight_system.nn.dimension_activations.items():
            dim_activations[str(dim)] = words[-100:]
        # 确保 learned_semantics 正确保存
        learned_semantics_to_save = {}
        for dim, info in self.weight_system.nn.learned_semantics.items():
            learned_semantics_to_save[str(dim)] = info        
        data = {
            'word_freq': dict(self.weight_system.word_freq),
            'bigram_freq': bigram_serializable,
            'sentence_patterns': self.weight_system.sentence_patterns,
            'correction_count': self.correction_count,
            'dialogue_history': self.dialogue_history[-100:],
            'big_groups': dict(self.weight_system.big_groups),
            'common_knowledge': self.weight_system.common_knowledge,
            'dictionary': self.weight_system.dictionary,
            'training_pairs': self.weight_system.training_pairs,
            'auto_learned_relations': self.weight_system.auto_learned_relations,
        }
        
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        torch.save({
            'model_state': self.weight_system.nn.state_dict(),
            'dimension_activations': dim_activations,
            'learned_semantics': self.weight_system.nn.learned_semantics,
        }, self.model_path)
        
        if not silent:
            print(f"[保存] 完成 - 维度激活: {len(dim_activations)}, 维度语义: {len(learned_semantics_to_save)}")
        
    def load_dictionary(self, dict_path="semantic_seed.txt"):
        if not os.path.exists(dict_path):
            print(f"[词典] 文件不存在: {dict_path}")
            return False
            
        print(f"[词典] 加载: {dict_path}")
        
        with open(dict_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        word_count = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split('|')
            if len(parts) >= 1:
                word = parts[0]
                category = parts[1] if len(parts) > 1 else ""
                emotion = float(parts[2]) if len(parts) > 2 else 0
                related = parts[3].split(',') if len(parts) > 3 else []
                
                self.weight_system.dictionary[word] = {
                    'category': category,
                    'emotion': emotion,
                    'related': related
                }
                
                self.weight_system.word_freq[word] += 1
                word_count += 1
                
                for r in related:
                    r = r.strip()
                    if r:
                        self.weight_system.set_common_knowledge(word, r, 0.9)
                        
        print(f"[词典] 加载完成: {word_count} 个词")
        return True
        
    def load_training_corpus(self, corpus_path="training_corpus.txt"):
        if not os.path.exists(corpus_path):
            print(f"[语料] 文件不存在: {corpus_path}")
            return 0
            
        print(f"[语料] 加载: {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        count = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if '>>>' in line:
                parts = line.split('>>>')
                if len(parts) == 2:
                    input_text = parts[0].strip()
                    output_text = parts[1].strip()
                    
                    self.train_from_input(input_text, silent=True)
                    self.train_from_input(output_text, silent=True)
                    self.weight_system.training_pairs.append((input_text, output_text))
                    count += 1
                    
        print(f"[语料] 加载完成: {count} 对话")
        return count
        
    def process_input(self, text):
        print(f"\n[输入] {text}")
        
        words = list(text)
        print(f"[分词] {words}")
        
        combinations = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            stat_ratio, nn_weight = self.weight_system.compute_combination_ratio(w1, w2)
            combinations.append({'combo': w1 + w2, 'stat_ratio': stat_ratio})
            
        combinations.sort(key=lambda x: x['stat_ratio'], reverse=True)
        print(f"[组合] {[c['combo'] for c in combinations[:3]]}")
        
        predictions = self._predict(words, combinations)
        print(f"[预测] {predictions}")
        
        best = self.validator.validate_and_iterate(words, predictions)
        print(f"[最终] {best}")
        
        self.dialogue_history.append({'input': text, 'output': best})
        return best
        
    def _predict(self, words, combinations):
        predictions = []
        idx_to_word = self.weight_system.idx_to_word
    
        # ========== 1. 获取输入权重 ==========
        input_weights = []
        for word in words:
            w = self.weight_system.get_word_weight(word)
            if w is not None:
                input_weights.append(w)
    
        if not input_weights:
            return ["嗯？", "什么？", "再说一遍？"]
    
        input_avg = torch.mean(torch.stack([w.squeeze(0) for w in input_weights]), dim=0)
    
        # ========== 2. 用神经网络找高关联词（核心！）==========
        high_relation = []
        for word, idx in self.weight_system.word_to_idx.items():
            if word in words:  # 跳过自己
                continue
        
            word_weight = self.weight_system.get_word_weight(word)
            if word_weight is None:
                continue
        
            # 神经网络计算关联分数
            score = self.weight_system.nn.compute_relation_score(
                input_avg, word_weight.squeeze(0)
            ).item()
        
            if score > 0.5:
                high_relation.append((word, score))
    
        # 按关联分数排序
        high_relation.sort(key=lambda x: x[1], reverse=True)
    
        # ========== 3. 基于高关联词生成回复 ==========
        if high_relation:
            top_word, top_score = high_relation[0]
        
            if top_score > 0.8:
                predictions.append(f"{top_word}！")
            predictions.append(f"你是说{top_word}吗？")
            predictions.append(f"跟{top_word}有关？")
        
            # 多个关联词
            if len(high_relation) >= 2:
                w2, _ = high_relation[1]
                predictions.append(f"{top_word}还是{w2}？")
    
        # ========== 4. 用 auto_learned_relations ==========
        for word in words:
            for key, (score, count) in self.weight_system.auto_learned_relations.items():
                if key.startswith(word + "→") and score > 0.5:
                    related = key.split("→")[1]
                    predictions.append(f"是不是{related}？")
    
        # ========== 5. 用 learned_semantics ==========
        learning_zone = DIMENSION_ZONES['learning']
        input_learning = input_avg[learning_zone[0]:learning_zone[1]]
    
        for dim, info in self.weight_system.nn.learned_semantics.items():
            dim_idx = dim - learning_zone[0]
            if 0 <= dim_idx < len(input_learning):
                if input_learning[dim_idx].abs().item() > 0.3:
                    for w in info.get('words', [])[:2]:
                        predictions.append(f"跟{w}有关？")
    
        # ========== 6. 兜底：训练对匹配 ==========
        input_str = ''.join(words)
        for pair_input, pair_output in self.weight_system.training_pairs:
            if pair_input in input_str or input_str in pair_input:
                predictions.append(pair_output)
    
        predictions = list(set(predictions))
        return predictions[:5] if predictions else ["嗯", "然后呢", "继续说"]
        
    def train_from_input(self, text, silent=False):
        words = list(text)
        for word in words:
            self.weight_system.word_freq[word] += 1
        for i in range(len(words) - 1):
            self.weight_system.bigram_freq[words[i]][words[i+1]] += 1
        self.weight_system.sentence_patterns.append(words)
        
        for word in words:
            if word in self.weight_system.word_to_idx:
                idx = torch.tensor([self.weight_system.word_to_idx[word]], device=DEVICE)
                self.weight_system.nn.track_dimension_activation(idx, word)
        
        if not silent:
            print(f"[训练] 学习: {text}")
    def _get_phrase_weight(self, phrase):
        """获取词语权重，支持单字和词语组合"""
        chars = list(phrase)
    
        if len(chars) == 0:
            return None
    
        # 单字直接获取
        if len(chars) == 1:
            return self.weight_system.get_word_weight(chars[0])
    
        # 词语：逐字组合计算
        idxs = [self.weight_system.word_to_idx.get(c) for c in chars]
        if None in idxs:
            return None
    
        # 第一个字
        idx_tensor = torch.tensor([idxs[0]], device=DEVICE)
        result_weight = self.weight_system.nn.get_combined_weight(idx_tensor)
    
        # 逐个组合
        for i in range(1, len(idxs)):
            idx_tensor2 = torch.tensor([idxs[i]], device=DEVICE)
            weight2 = self.weight_system.nn.get_combined_weight(idx_tensor2)
        
            # 计算组合权重
            combined = (result_weight + weight2) / 2
            result_weight = combined
    
        return result_weight

        
    def auto_train(self, epochs=10, show_progress=True):
        print(f"\n[主训练] {epochs} 轮...")
    
        optimizer = torch.optim.Adam(self.weight_system.nn.parameters(), lr=LEARNING_RATE)
        progress = ProgressDisplay(epochs, "主训练")
    
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            batch_count = 0
            meta_activations = []
        
            # ========== 主训练循环 ==========
            for p_idx, pattern in enumerate(self.weight_system.sentence_patterns):
                if len(pattern) < 2:
                    continue
                for i in range(len(pattern) - 1):
                    w1, w2 = pattern[i], pattern[i+1]
                    if w1 not in self.weight_system.word_to_idx:
                        continue
                    if w2 not in self.weight_system.word_to_idx:
                        continue
                    
                    optimizer.zero_grad()
                
                    idx1 = torch.tensor([self.weight_system.word_to_idx[w1]], device=DEVICE)
                    idx2 = torch.tensor([self.weight_system.word_to_idx[w2]], device=DEVICE)
                
                    combo = self.weight_system.nn.compute_combination_weight(idx1, idx2)
                
                    # ========== 元学习区闭环管理 ==========
                    learning_zone = DIMENSION_ZONES['learning']
                    meta_zone = DIMENSION_ZONES['meta_learning']
                
                    learning_weights = combo[0, learning_zone[0]:learning_zone[1]]
                
                    BASE_MEAN = 0.5
                    BASE_STD = 6.0
                    BASE_ACTIVE = 0.1
                
                    meta_signal = combo[0, meta_zone[0]:meta_zone[1]]
                    scale = meta_signal.mean()
                
                    modulated_learning = learning_weights * scale
                
                    combo_modulated = combo.clone()
                    start, end = learning_zone[0], learning_zone[1]
                    combo_modulated = torch.cat([
                        combo[:, :start],
                        modulated_learning.unsqueeze(0),
                        combo[:, end:]
                    ], dim=1)
                
                    meta_zone = DIMENSION_ZONES['meta_learning']
                    meta_act = combo_modulated[0, meta_zone[0]:meta_zone[1]].detach()
                    meta_activations.append(meta_act)
                
                    bigram_count = self.weight_system.bigram_freq[w1][w2]
                    w1_total = sum(self.weight_system.bigram_freq[w1].values())
                    stat_prob = bigram_count / w1_total if w1_total > 0 else 0.05
                
                    target = torch.zeros_like(combo)
                    target[0, 0] = min(stat_prob, 1.0)
                    target[0, 1:32] = min(stat_prob * 0.3, 0.5)
                    target[0, 32:64] = torch.randn(32, device=DEVICE) * 0.05
                    target[0, 64:3564] = torch.zeros(3500, device=DEVICE)
                    target[0, 3564:4080] = torch.zeros(516, device=DEVICE)
                
                    learning_zone = DIMENSION_ZONES['learning']
                    loss = torch.nn.functional.mse_loss(combo_modulated[:, :learning_zone[1]], target[:, :learning_zone[1]])
                
                    loss.backward()
                    optimizer.step()
                
                    self.weight_system.nn.track_dimension_activation(idx1, w1)
                    self.weight_system.nn.track_dimension_activation(idx2, w2)
                
                    total_loss += loss.item()
                    batch_count += 1
                
                    self.weight_system.speed_controller.control('batch')
        
            # ========== 关联训练（每个epoch结束后）==========
            relation_total = 0
            relation_count = 0
        
            for word, relations in self.weight_system.common_knowledge.items():
                # 检查word的每个字符是否都在词表里
                word_chars = list(word)
                if not all(c in self.weight_system.word_to_idx for c in word_chars):
                    continue
    
                for related_word, target_score in relations.items():
                    # 检查related_word的每个字符是否都在词表里
                    related_chars = list(related_word)
                    if not all(c in self.weight_system.word_to_idx for c in related_chars):
                        continue
        
                    optimizer.zero_grad()
        
                    # 获取word的权重（支持单字和词语）
                    word_weight = self._get_phrase_weight(word)
        
                    # 获取related_word的权重（支持单字和词语）
                    related_weight = self._get_phrase_weight(related_word)
        
                    if word_weight is None or related_weight is None:
                        continue
        
                    current = self.weight_system.nn.compute_relation_score(word_weight, related_weight)
                    target = torch.tensor([[target_score]], device=DEVICE)
        
                    relation_loss = torch.nn.functional.mse_loss(current, target)
        
                    relation_loss.backward()
                    optimizer.step()
        
                    relation_total += relation_loss.item()
                    relation_count += 1
        
            # ========== 更新进度（只执行一次）==========
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            avg_relation_loss = relation_total / relation_count if relation_count > 0 else 0
            epoch_time = time.time() - epoch_start
            progress.update(epoch, avg_loss, f"主:{batch_count} 关联:{relation_count}")
            progress.epoch_done(epoch_time)
        
            # ========== 元认知监控（只输出一次）==========
            if meta_activations:
                meta_stack = torch.stack(meta_activations)
                meta_mean = meta_stack.mean().item()
                meta_std = meta_stack.std().item()
                meta_max = meta_stack.max().item()
                active_dims = (meta_stack.abs() > 0.1).any(dim=0).sum().item()
                print(f"\n[元认知] 均值:{meta_mean:.4f} 标准差:{meta_std:.4f} 最大:{meta_max:.4f} 活跃维:{active_dims}/32 | 关联Loss:{avg_relation_loss:.4f}")
        
            self.weight_system.speed_controller.control('epoch')
            
        progress.finish()
        
    def infer_dimension_semantics(self):
        learned = self.weight_system.nn.infer_dimension_semantics()
        
        print("\n" + "=" * 60)
        print("[维度语义报告]")
        print("=" * 60)
        
        zones = DIMENSION_ZONES
        
        print(f"\n预定义维度 ({zones['predefined'][0]}-{zones['predefined'][1]-1}):")
        for dim in range(zones['predefined'][0], zones['predefined'][1]):
            label = SEMANTIC_LABELS.get(dim, f"维度{dim}")
            print(f"  维度{dim}: {label}")
       
        print(f"\n常识知识维度 ({zones['knowledge'][0]}-{zones['knowledge'][1]-1}):")
        for dim, info in sorted(learned.items()):
            if zones['knowledge'][0] <= dim < zones['knowledge'][1]:
                print(f"  维度{dim}: {info['inferred_meaning']}")
       
        print(f"\n自主学习维度 ({zones['learning'][0]}-{zones['learning'][1]-1}):")
        learned_in_learning = {dim: info for dim, info in learned.items() 
                               if zones['learning'][0] <= dim < zones['learning'][1]}
        if learned_in_learning:
            for dim, info in sorted(learned_in_learning.items())[:10]:
                print(f"  维度{dim}: {info['inferred_meaning']}")
            if len(learned_in_learning) > 10:
                print(f"  ... 共 {len(learned_in_learning)} 个维度已学习")
        else:
            print("  尚未学习")
       
        print(f"\n缓冲区维度 ({zones['buffer'][0]}-{zones['buffer'][1]-1}): 用于分担计算压力")
       
        print("=" * 60)
        def discover_new_relations(self, threshold=0.7):
            """神经网络自己发现新关联"""
            print("\n[发现] 探索新关联...")
    
            new_found = []
    
            for w1, idx1 in self.weight_system.word_to_idx.items():
                w1_weight = self.weight_system.get_word_weight(w1)
                if w1_weight is None:
                    continue
        
            for w2, idx2 in self.weight_system.word_to_idx.items():
                if w1 == w2:
                    continue
                if w2 in self.weight_system.common_knowledge.get(w1, {}):
                    continue
            
                w2_weight = self.weight_system.get_word_weight(w2)
                if w2_weight is None:
                    continue
            
                score = self.weight_system.nn.compute_relation_score(
                    w1_weight.squeeze(0), w2_weight.squeeze(0)
                ).item()
            
                if score >= threshold:
                    new_found.append((w1, w2, score))
        new_found = []
        new_found.sort(key=lambda x: x[2], reverse=True)
    
        for w1, w2, score in new_found[:10]:
            print(f"  发现: {w1} → {w2} = {score:.3f}")
            self.weight_system.auto_learn_relation(w1, w2, score)
    
        return new_found[:10]
    def self_dialogue(self, turns=3):
        print(f"\n[自我对话] {turns}轮...")
        starters = ["我饿了", "我困了", "我难过了", "我开心了", "我累了", "我生气了", "我很孤独", "我很焦虑"]
        
        for i in range(turns):
            starter = random.choice(starters)
            print(f"\n--- 第{i+1}轮 ---")
            print(f"[A] {starter}")
            response = self.process_input(starter)
            print(f"[B] {response}")
            self.train_from_input(starter)
            self.train_from_input(response)
            
        print("[自我对话] 完成")
        
    def show_status(self):
        print("\n" + "=" * 60)
        print("[系统状态]")
        print("=" * 60)
        print(f"设备: {DEVICE}")
        if DEVICE.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem_used = torch.cuda.memory_allocated(0) / 1024**2
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"GPU内存: {mem_used:.1f}MB / {mem_total:.1f}MB")
        print(f"向量维度: {self.vector_dim}")
        print(f"词表大小: {len(self.weight_system.word_to_idx)}")
        print(f"训练语料: {len(self.weight_system.training_pairs)} 对")
        print(f"常识关联: {len(self.weight_system.common_knowledge)} 条")
        print(f"自动学习关联: {len(self.weight_system.auto_learned_relations)} 条")
        print(f"维度语义已学习: {len(self.weight_system.nn.learned_semantics)} 个")
        print(f"对话历史: {len(self.dialogue_history)} 条")
        print(f"速度控制: {'开启' if SPEED_CONTROL_ENABLED else '关闭'}")
        print(f"自动保存间隔: {AUTO_SAVE_INTERVAL} 秒")
        print("=" * 60)
        
    def run(self):
        print("\n" + "=" * 60)
        print("智能对话系统 v5.0- 4096维GPU加速版")
        print("命令: exit(退出) train(训练) save(保存) self(自我对话)")
        print("      status(状态) semantics(维度语义) speed[1-10](调速度)")
        print("=" * 60)
        
        self.auto_saver = AutoSaveManager(self)
        self.auto_saver.start()
        
        try:
            while True:
                try:
                    user_input = input("\n你: ").strip()
                    
                    if not user_input:
                        continue
                        
                    if user_input.lower() == 'exit':
                        self.save_data()
                        print("再见！")
                        break
                        
                    elif user_input.lower() == 'train':
                        self.auto_train(epochs=20)
                        self.discover_new_relations(threshold=0.6)
                        
                    elif user_input.lower().startswith('train '):
                        try:
                            epochs = int(user_input[6:].strip())
                            if epochs > 0:
                                self.auto_train(epochs=epochs)
                            else:
                                print("[训练] 请输入正整数")
                        except:
                            print("[训练] 用法: train 200") 
                        
                    elif user_input.lower() == 'save':
                        self.save_data()
                        
                    elif user_input.lower() == 'self':
                        self.self_dialogue(turns=2)
                        
                    elif user_input.lower() == 'status':
                        self.show_status()
                    elif user_input.lower() == 'test':
                        self.run_all_tests()
                        
                    elif user_input.lower() == 'semantics':
                        self.infer_dimension_semantics()
                        
                    elif user_input.lower().startswith('speed'):
                        try:
                            level = int(user_input[5:].strip())
                            if 1 <= level <= 10:
                                self.weight_system.speed_controller.adjust_speed(level)
                                print(f"[速度] 已调整为 {level} 级")
                            else:
                                print("[速度] 请输入 1-10 的数字")
                        except:
                            print("[速度] 用法: speed 5")
                            
                    else:
                        response = self.process_input(user_input)
                        print(f"\nAI: {response}")
                        self.train_from_input(user_input)
                        
                except KeyboardInterrupt:
                    print("\n\n[中断] 保存数据...")
                    self.save_data()
                    break
        finally:
            self.auto_saver.stop()


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("智能对话系统 v5.0 - 4096维GPU加速 + 维度语义学习")
    print("=" * 60)
    
    MODEL_PATH = "chat_v5.0_model.pt"
    DATA_PATH = "chat_v5.0_data.json"
    
    chat = ChatSystem(vector_dim=VECTOR_DIM, model_path=MODEL_PATH, data_path=DATA_PATH)
    
    # ========== 1. 先收集所有字符 ==========
    all_chars = set()
    
    if os.path.exists("semantic_seed.txt"):
        with open("semantic_seed.txt", "r", encoding="utf-8") as f:
            all_chars.update(f.read())
    
    if os.path.exists("training_corpus.txt"):
        with open("training_corpus.txt", "r", encoding="utf-8") as f:
            all_chars.update(f.read())
    
    print(f"[字符集] 收集到 {len(all_chars)} 个唯一字符")
    
    # ========== 2. 一次性构建词表 ==========
    chat.weight_system.build_vocab(list(all_chars))
    chat.weight_system.compute_statistical_weights()
    
    # ========== 3. 加载词典和语料 ==========
    chat.load_dictionary("semantic_seed.txt")
    chat.load_training_corpus("training_corpus.txt")
    
    # ========== 4. 设置大组和常识 ==========
    print("\n[大组] 形成语义大组...")
    chat.weight_system.form_big_group("饿", ["我饿", "饿了"])
    chat.weight_system.form_big_group("困", ["我困", "困了"])
    chat.weight_system.form_big_group("难过", ["难过", "过了"])
    chat.weight_system.form_big_group("开心", ["开心", "心了"])
    chat.weight_system.form_big_group("累", ["我累", "累了"])
    chat.weight_system.form_big_group("生气", ["生气", "气了"])
    
    print("\n[常识] 设置固定常识权重...")
    chat.weight_system.set_common_knowledge("饿", "吃", 0.95)
    chat.weight_system.set_common_knowledge("困", "睡", 0.95)
    chat.weight_system.set_common_knowledge("人", "生物", 0.95)
    chat.weight_system.set_common_knowledge("累", "休息", 0.95)
    
    # ========== 5. 检查是否需要训练 ==========
    SKIP_TRAINING = os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH)
    
    vocab_size = len(chat.weight_system.word_to_idx)
    base_epochs = max(50, vocab_size // 5)
    print(f"\n[训练配置] 词表大小: {vocab_size}, 基础训练轮次: {base_epochs}")
    
    if SKIP_TRAINING:
        print("\n[跳过] 检测到已保存的模型，跳过初始训练")
    else:
        print("\n[训练] 未检测到模型，开始初始训练...")
        
        print("\n" + "=" * 60)
        print("[阶段1] 常识预训练")
        print("=" * 60)
        chat.weight_system.apply_common_knowledge_training(epochs=10)
        
        print("\n" + "=" * 60)
        print("[阶段2] 主训练")
        print("=" * 60)
        chat.auto_train(epochs=10)
        chat.infer_dimension_semantics()
    
    print("\n" + "=" * 60)
    print("[自我对话测试]")
    print("=" * 60)
    chat.self_dialogue(turns=3)
    
    print("\n" + "=" * 60)
    chat.save_data()
    print("=" * 60)
    
    chat.show_status()
    
    chat.run()