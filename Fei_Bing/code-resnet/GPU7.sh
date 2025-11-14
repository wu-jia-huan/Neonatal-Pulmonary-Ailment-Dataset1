#原始版本全监督
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D_yrh_resnet50_Xi_Ru_ViT.py --root_path ../data/ACDC --exp Complexity/Fully_ViT_num --num_classes 2 --labeled_num 140 --batch_size 8 --max_iterations 3000

CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D_yrh_resnet50_Xi_Ru.py --root_path ../data/ACDC --exp Complexity/Xi_Ru_DensNet  --num_classes 2 --labeled_num 140 --batch_size 8 --max_iterations 3000

CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D_yrh_resnet50_Fei_Tou.py --root_path ../data/ACDC --exp Complexity/Fei_Tou_DensNet --num_classes 5 --labeled_num 140 --batch_size 4 --max_iterations 3000


#废弃
CUDA_VISIBLE_DEVICES=5 python train_cross_pesudo_resent50_xi_ru.py --root_path ../data/ACDC --exp cross_pesudo_Xi_Ru/Fully_DenseNet50_num --num_classes 2 --labeled_num 140 --batch_size 4 --max_iterations 3000
CUDA_VISIBLE_DEVICES=1 python train_cross_pesudo_resent50_xi_ru_vit.py --root_path ../data/ACDC --exp cross_pesudo_Xi_Ru_vit/Fully_vit_num --num_classes 2 --labeled_num 140 --batch_size 8 --max_iterations 3000
CUDA_VISIBLE_DEVICES=6 python train_cross_pesudo_resent50_fei_tou.py --root_path ../data/ACDC --exp cross_pesudo_fei_tou/Fully_DenseNet50_num --num_classes 5 --labeled_num 140 --batch_size 4 --max_iterations 3000


#最终版本污染CutMix
CUDA_VISIBLE_DEVICES=0 python train_cross_pesudo_resent50_xi_ru_vit2.py --root_path ../data/ACDC --exp Complexity/Xi_Ru --num_classes 2 --labeled_num 140 --batch_size 8 --max_iterations 3000

CUDA_VISIBLE_DEVICES=7 python train_cross_pesudo_resent50_xi_ru_vit_true.py --root_path ../data/ACDC --exp cross_pesudo_Xi_Ru_vit_true/Fully_ViT_num --num_classes 2 --labeled_num 140 --batch_size 16 --max_iterations 20

CUDA_VISIBLE_DEVICES=0 python train_cross_pesudo_resent50_fei_tou2.py --root_path ../data/ACDC --exp Complexity/Fei_Tou --num_classes 5 --labeled_num 140 --batch_size 4 --max_iterations 3000


#2023MICCAI action++复现
CUDA_VISIBLE_DEVICES=3 python train_fully_supervised_Xi_Ru_action++.py --root_path ../data/ACDC --exp xi_ru_action++/xi_ru --num_classes 2 --labeled_num 140 --batch_size 4 --max_iterations 3

CUDA_VISIBLE_DEVICES=3 python train_fully_supervised_Fei_tou_action++.py --root_path ../data/ACDC --exp Fei_Tou_action++/Fei_Tou --num_classes 5 --labeled_num 140 --batch_size 4 --max_iterations 3000


#补充可视化
CUDA_VISIBLE_DEVICES=2 python train_Xi_Ru_visualization.py --root_path ../data/ACDC --exp Xi_Ru_visualization/Xi_Ru_uncertainty --num_classes 2 --labeled_num 140 --batch_size 8 --max_iterations 3000

CUDA_VISIBLE_DEVICES=2 python train_Fei_Tou_visualization.py --root_path ../data/ACDC --exp Fei_Tou_visualization/Fei_Tou_uncertainty --num_classes 5 --labeled_num 140 --batch_size 4 --max_iterations 3000


#增加文本信息
CUDA_VISIBLE_DEVICES=7 python train_Xi_Ru.py --root_path ../data/ACDC --exp Xi_Ru_text/Xi_Ru --num_classes 2 --labeled_num 140 --batch_size 4 --max_iterations 3000

CUDA_VISIBLE_DEVICES=2 python train_Fei_Tou.py --root_path ../data/ACDC --exp Fei_Tou_text/Fei_Tou --num_classes 5 --labeled_num 140 --batch_size 4 --max_iterations 3

#增加指标AUROC
CUDA_VISIBLE_DEVICES=0 python train_xi_ru_AUROC.py --root_path ../data/ACDC --exp AUROC/Xi_Ru --num_classes 2 --labeled_num 140 --batch_size 8 --max_iterations 3

CUDA_VISIBLE_DEVICES=0 python train_xi_ru_vit_AUROC.py --root_path ../data/ACDC --exp AUROC/Xi_Ru_ViT --num_classes 2 --labeled_num 140 --batch_size 8 --max_iterations 3000

CUDA_VISIBLE_DEVICES=0 python train_fully_xi_ru_AUROC.py --root_path ../data/ACDC --exp AUROC/DenseNet --num_classes 2 --labeled_num 140 --batch_size 8 --max_iterations 3000

#k折交叉验证
CUDA_VISIBLE_DEVICES=0 python train_Xi_ru_kfold.py --root_path ../data/ACDC --exp k-fold/Xi_Ru --num_classes 2 --labeled_num 140 --batch_size 8 --max_iterations 3000

CUDA_VISIBLE_DEVICES=0 python train_Fei_Tou_kfold.py --root_path ../data/ACDC --exp k-fold/Fei_Tou --num_classes 5 --labeled_num 140 --batch_size 4 --max_iterations 3000

python analyze_kfold_results.py