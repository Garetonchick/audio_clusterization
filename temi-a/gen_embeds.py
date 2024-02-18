import os
import torch
import argparse
import torch.nn.functional as F

def gen_normalized_embeds(encoder, dataset):
    raise NotImplementedError()

def get_dataset(dataset_name):
    raise NotImplementedError()

def get_encoder(encoder_name, dataset):
    raise NotImplementedError()

def calc_knn(embeds, k):
    # Naive implementation
    sims = embeds @ embeds.permute(1, 0)
    sims.fill_diagonal_(float('-inf'))
    _, indicies = torch.topk(sims, k=k, dim=1)
    return indicies

def main(args):
    embeds_name = f"{args.dataset}-{args.encoder}-embeds.pth"
    knn_name = f"{args.dataset}-{args.encoder}-knn.pth"
    embeds = None
    if not args.only_knn:
        dataset = get_dataset(args.dataset)
        encoder = get_encoder(args.encoder, dataset)
        embeds = gen_normalized_embeds(encoder, dataset)
    else:
        embeds = torch.load(os.path.join(args.data_dir, embeds_name))
        embeds = F.normalize(embeds, p=2, dim=1) 
    
    knn_indicies = calc_knn(embeds, embeds.shape[0] // args.n_clusters)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(embeds, os.path.join(args.out_dir, embeds_name))
    torch.save(knn_indicies, os.path.join(args.out_dir, knn_name))

description = \
"""
Generates embeddings from audio
using specified pretrained encoder model. Calculates 
KNN for each embedding in feature space.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='temi_gen_embeds',
                    description=description 
    )
    parser.add_argument(
        '--data_dir', default="data", type=str,
        help="Path to your audio dataset/embeddings"
    )
    parser.add_argument(
        '--out_dir', default="out", type=str,
        help="Path to output embeddings"
    )
    parser.add_argument(
        '--dataset', default="DCASE2018_TASK5", choices=["DCASE2018_TASK5"], type=str,
        help="Dataset to generate embeddings from"
    )
    parser.add_argument(
        '--encoder', default="PaSST", choices=["PaSST"], type=str,
        help="Encoder used to generate embeddings from audio"
    )
    parser.add_argument(
        '--only_knn', default=False, action="store_true",
        help="""Generate KNN for embeddings in data_dir. 
        Embeddings must have name in format \"{dataset}-{encoder}-embeds.pth\""""
    )
    parser.add_argument(
        '--n_clusters', required=True, type=int,
        help="Number of clusters. Used to calculate K=N/n_clusters for KNN"
    )
    main(parser.parse_args())