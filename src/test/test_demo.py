#!/usr/bin/env python3
"""
全面测试demo样本的检索性能，使用BoB技术
"""
import torch
import os
import sys
import csv
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from bitarray import util as butil
import bitarray

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from dataset.DemoDataset import DemoDataset, LABEL_NAMES
from model.PathSearch import PathSearch


class BoB:
    """Bag-of-Barcodes representation for pathology images"""
    def __init__(self, barcodes, semantic_embedding, name, site):
        self.barcodes = [bitarray.bitarray(b.tolist()) for b in barcodes]
        self.name = name
        self.site = site
        self.semantic_embedding = semantic_embedding

    def distance(self, bob):
        """Compute distance between two BoB representations"""
        total_dist = []
        for feat in self.barcodes:
            distances = [butil.count_xor(feat, b) for b in bob.barcodes]
            total_dist.append(np.min(distances))
        retval = np.median(total_dist)
        semantic_dist = torch.norm(self.semantic_embedding - bob.semantic_embedding, p=2, dim=1)
        return retval + 200 * semantic_dist.item()

def test_retrieval(model, dataset, device, top_k=5):
    """测试检索性能，使用BoB技术"""
    model.eval()
    
    # 编码所有样本
    print("Encoding all samples...")
    all_img_feats = []
    all_mosaics = []
    all_labels = []
    all_case_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Encoding"):
            case_id, data, label, _, _ = dataset[i]
            data = data.to(device).unsqueeze(0)
            
            # 通过模型获取特征和mosaic
            img_feat, mosaic = model.encode_image(data)
            
            all_case_ids.append(case_id)
            all_img_feats.append(img_feat.cpu())
            all_labels.append(label)
            
            # Combine mosaic and image feature
            mosaic_combined = torch.cat((mosaic.cpu().squeeze(0), img_feat.cpu()), dim=0)
            all_mosaics.append(mosaic_combined)
    
    # 构建BoB表示
    print("\nBuilding BoB representations...")
    BoBs = {}
    for i in range(len(all_mosaics)):
        mosaic_feat = all_mosaics[i]
        # Convert to barcodes using derivative
        barcodes = (np.diff(np.array(mosaic_feat), n=1, axis=1) < 0) * 1
        BoBs[all_case_ids[i]] = BoB(barcodes, all_img_feats[i], all_case_ids[i], all_labels[i])
    
    # 计算距离
    print("\nComputing distances...")
    distances = defaultdict(list)
    for a in tqdm(BoBs, desc="Computing distances"):
        for b in BoBs:
            if a == b:
                continue
            dist = BoBs[a].distance(BoBs[b])
            distances[a].append((dist, b))
    
    # 排序
    for query in distances:
        distances[query].sort(key=lambda x: x[0])
    
    # 对每个样本进行检索评估
    print("\nEvaluating retrieval results...")
    results = []
    correct_top1 = 0
    correct_top3_mv = 0
    correct_top5_mv = 0
    
    for query_case in all_case_ids:
        query_label = BoBs[query_case].site
        query_class = LABEL_NAMES.get(query_label, 'UNKNOWN')
        
        # Get top-k results
        top_results = distances[query_case][:top_k]
        
        # Top-1
        top1_label = BoBs[top_results[0][1]].site
        if top1_label == query_label:
            correct_top1 += 1
        
        # Top-3 Majority Vote
        top3_labels = [BoBs[top_results[i][1]].site for i in range(min(3, len(top_results)))]
        top3_mv_label = Counter(top3_labels).most_common(1)[0][0]
        if top3_mv_label == query_label:
            correct_top3_mv += 1
        
        # Top-5 Majority Vote
        top5_labels = [BoBs[top_results[i][1]].site for i in range(min(5, len(top_results)))]
        top5_mv_label = Counter(top5_labels).most_common(1)[0][0]
        if top5_mv_label == query_label:
            correct_top5_mv += 1
        
        # Save detailed results
        for rank, (dist, retrieved_case) in enumerate(top_results, 1):
            retrieved_label = BoBs[retrieved_case].site
            retrieved_class = LABEL_NAMES.get(retrieved_label, 'UNKNOWN')
            match = '✓' if query_label == retrieved_label else '✗'
            
            results.append({
                'Query_Case': query_case,
                'Query_Class': query_class,
                'Rank': rank,
                'Retrieved_Case': retrieved_case,
                'Retrieved_Class': retrieved_class,
                'Distance': f"{dist:.2f}",
                'Match': match
            })
    
    return results, correct_top1, correct_top3_mv, correct_top5_mv, len(all_case_ids)


def save_results(results, output_file):
    """保存结果到CSV"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Query_Case', 'Query_Class', 'Rank',
            'Retrieved_Case', 'Retrieved_Class', 'Distance', 'Match'
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {output_file}")


def print_summary(results, correct_top1, correct_top3_mv, correct_top5_mv, total_queries):
    """打印统计摘要"""
    print("\n" + "=" * 60)
    print("RETRIEVAL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total queries: {total_queries}")
    print(f"Top-1 Accuracy: {correct_top1}/{total_queries} = {correct_top1/total_queries*100:.2f}%")
    print(f"Top-3 MV Accuracy: {correct_top3_mv}/{total_queries} = {correct_top3_mv/total_queries*100:.2f}%")
    print(f"Top-5 MV Accuracy: {correct_top5_mv}/{total_queries} = {correct_top5_mv/total_queries*100:.2f}%")
    
    # 按类别统计
    print("\nPer-class Top-1 performance:")
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for result in results:
        if result['Rank'] == 1:
            cls = result['Query_Class']
            class_stats[cls]['total'] += 1
            if result['Match'] == '✓':
                class_stats[cls]['correct'] += 1
    
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {cls}: {stats['correct']}/{stats['total']} = {acc:.2f}%")
    
    print("\nFailed queries (Top-1 incorrect):")
    for result in results:
        if result['Rank'] == 1 and result['Match'] == '✗':
            # Shorten case IDs for display
            query_short = '-'.join(result['Query_Case'].split('-')[:3])
            retrieved_short = '-'.join(result['Retrieved_Case'].split('-')[:3])
            print(f"  {result['Query_Class']} ({query_short}): "
                  f"Retrieved {result['Retrieved_Class']} ({retrieved_short}) (dist={result['Distance']})")


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='PathSearch Demo Retrieval Test')
    parser.add_argument('--device', default='cpu', type=str, help='Device to use (cpu or cuda)')
    parser.add_argument('--data_dir', type=str, default='./demo_dataset', help='Path to demo dataset')
    parser.add_argument('--model_path', type=str, default='', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./demo_retrieval_results.csv', help='Output CSV file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PathSearch Demo Retrieval Test (BoB)")
    print("=" * 60)
    
    # 配置
    device = torch.device(args.device)
    data_dir = args.data_dir
    output_file = args.output
    model_path = args.model_path
    
    print(f"\nDevice: {device}")
    print(f"Data directory: {data_dir}")
    
    # 加载数据集
    print("\nLoading dataset...")
    dataset = DemoDataset(path=data_dir, mode='test', sample_num=None)
    print(f"Loaded {len(dataset)} samples")
    
    # 加载模型
    print("\nInitializing model...")
    model = PathSearch(768).to(device)
    
    if model_path and os.path.isfile(model_path):
        print(f"Loading model from: {model_path}")
        state = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = state['model'] if 'model' in state else state
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully!")
    else:
        print("Using model with random weights for testing")
    
    # 测试检索
    results, correct_top1, correct_top3_mv, correct_top5_mv, total_queries = test_retrieval(
        model, dataset, device, top_k=5
    )
    
    # 保存结果
    save_results(results, output_file)
    
    # 打印摘要
    print_summary(results, correct_top1, correct_top3_mv, correct_top5_mv, total_queries)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
