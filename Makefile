# EE303_Project Makefile
# 用於管理兩階段商品匹配系統的所有工具

.PHONY: help install validate test clean data-processing dataset-generation model-training model-testing pipeline

# 預設目標
help:
	@echo "EE303_Project 工具管理"
	@echo "======================"
	@echo ""
	@echo "可用命令:"
	@echo "  make help                    - 顯示此幫助訊息"
	@echo "  make install                 - 安裝依賴套件"
	@echo "  make validate                - 驗證所有工具"
	@echo "  make test                    - 運行基本功能測試"
	@echo "  make clean                   - 清理暫存檔案"
	@echo ""
	@echo "資料處理:"
	@echo "  make data-processing         - 處理 MySQL 資料"
	@echo "  make view-data               - 檢視處理後的資料"
	@echo ""
	@echo "資料集生成:"
	@echo "  make dataset-generation      - 生成訓練資料集"
	@echo "  make bm25-samples            - 生成 BM25 樣本"
	@echo "  make classification-dataset  - 生成分類資料集"
	@echo "  make triplet-dataset         - 生成 Triplet 資料集"
	@echo ""
	@echo "模型訓練:"
	@echo "  make model-training          - 訓練所有模型"
	@echo "  make train-triplet           - 訓練 Triplet 模型"
	@echo "  make train-classification    - 訓練分類模型"
	@echo ""
	@echo "模型測試:"
	@echo "  make model-testing           - 測試所有模型"
	@echo "  make test-similarity         - 測試相似度模型"
	@echo "  make test-classification     - 測試分類模型"
	@echo "  make test-two-stage          - 測試兩階段模型"
	@echo ""
	@echo "完整管道:"
	@echo "  make pipeline                - 執行完整訓練和測試管道"
	@echo ""
	@echo "範例用法:"
	@echo "  make dataset-generation LEAF_FILE=data/leaf.csv ROOT_FILE=data/root.csv"
	@echo "  make train-triplet DATA_PATH=training_data/triplet.csv"
	@echo "  make pipeline TRAINING_DATA_DIR=training_data"

# 安裝依賴
install:
	@echo "安裝依賴套件..."
	pip install -r requirements.txt

# 驗證工具
validate:
	@echo "驗證所有工具..."
	cd tools && PYTHONPATH=.. python quick_validate.py

# 基本功能測試
test:
	@echo "運行基本功能測試..."
	cd tools && PYTHONPATH=.. python test_basic_functionality.py

# 清理暫存檔案
clean:
	@echo "清理暫存檔案..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name "validation_results.json" -delete
	rm -rf test_data/
	rm -rf training_data/
	rm -rf pipeline_results/

# 資料處理
data-processing:
	@echo "處理 MySQL 資料..."
	cd tools/data_processing && PYTHONPATH=../.. python simple_correct_parser.py

view-data:
	@echo "檢視處理後的資料..."
	cd tools/data_processing && PYTHONPATH=../.. python view_correct_data.py

# 資料集生成
dataset-generation: bm25-samples classification-dataset triplet-dataset
	@echo "所有資料集生成完成！"

bm25-samples:
	@echo "生成 BM25 樣本..."
	cd tools/dataset_generators && PYTHONPATH=../.. python bm25_sampler.py \
		--leaf_file ../../data/correct_structure_data/leaf_v1_correct.csv \
		--root_file ../../data/correct_structure_data/root_v1_correct.csv \
		--output samples.json \
		--k 5 \
		--sample_size 1000

classification-dataset:
	@echo "生成分類資料集..."
	cd tools/dataset_generators && PYTHONPATH=../.. python classification_dataset_generator.py \
		--samples_file samples.json \
		--leaf_file ../../data/correct_structure_data/leaf_v1_correct.csv \
		--root_file ../../data/correct_structure_data/root_v1_correct.csv \
		--output classification_dataset.csv \
		--embedding_dim 512 \
		--use_clip

triplet-dataset:
	@echo "生成 Triplet 資料集..."
	cd tools/dataset_generators && PYTHONPATH=../.. python triplet_dataset_generator.py \
		--samples_file samples.json \
		--leaf_file ../../data/correct_structure_data/leaf_v1_correct.csv \
		--root_file ../../data/correct_structure_data/root_v1_correct.csv \
		--output triplet_dataset.csv \
		--embedding_dim 512 \
		--triplet_per_anchor 3 \
		--max_triplets 10000 \
		--use_clip

# 模型訓練
model-training: train-triplet train-classification
	@echo "所有模型訓練完成！"

train-triplet:
	@echo "訓練 Triplet 模型..."
	cd tools/model_training && PYTHONPATH=../.. python train_triplet_model.py \
		--data_path ../../tools/dataset_generators/triplet_dataset.csv \
		--output_dir triplet_model \
		--text_embedding_dim 512 \
		--image_embedding_dim 512 \
		--hidden_dim 256 \
		--output_dim 128 \
		--batch_size 32 \
		--epochs 50 \
		--learning_rate 0.001 \
		--device auto

train-classification:
	@echo "訓練分類模型..."
	cd tools/model_training && PYTHONPATH=../.. python train_classification_model.py \
		--data_path ../../tools/dataset_generators/classification_dataset.csv \
		--output_dir classification_model \
		--text_embedding_dim 512 \
		--image_embedding_dim 512 \
		--hidden_dim 256 \
		--batch_size 32 \
		--epochs 50 \
		--learning_rate 0.001 \
		--device auto

# 模型測試
model-testing: test-similarity test-classification test-two-stage
	@echo "所有模型測試完成！"

test-similarity:
	@echo "測試相似度模型..."
	cd tools/model_testing && PYTHONPATH=../.. python test_models.py \
		--test_data_path ../../tools/dataset_generators/triplet_dataset.csv \
		--similarity_model_path ../../tools/model_training/triplet_model/best_triplet_model.pth \
		--output_dir test_results \
		--batch_size 32 \
		--device auto

test-classification:
	@echo "測試分類模型..."
	cd tools/model_testing && PYTHONPATH=../.. python test_models.py \
		--test_data_path ../../tools/dataset_generators/classification_dataset.csv \
		--classification_model_path ../../tools/model_training/classification_model/best_classification_model.pth \
		--output_dir test_results \
		--batch_size 32 \
		--device auto

test-two-stage:
	@echo "測試兩階段組合模型..."
	cd tools/model_testing && PYTHONPATH=../.. python test_models.py \
		--test_data_path ../../tools/dataset_generators/classification_dataset.csv \
		--similarity_model_path ../../tools/model_training/triplet_model/best_triplet_model.pth \
		--classification_model_path ../../tools/model_training/classification_model/best_classification_model.pth \
		--output_dir test_results \
		--similarity_threshold 0.5 \
		--batch_size 32 \
		--device auto

# 完整管道
pipeline:
	@echo "執行完整訓練和測試管道..."
	cd tools/pipelines && PYTHONPATH=../.. python train_and_test_pipeline.py \
		--training_data_dir ../../tools/dataset_generators/training_data \
		--output_dir ../../pipeline_results \
		--text_embedding_dim 512 \
		--image_embedding_dim 512 \
		--hidden_dim 256 \
		--output_dim 128 \
		--batch_size 32 \
		--epochs 50 \
		--learning_rate 0.001 \
		--similarity_threshold 0.5 \
		--device auto

# 自定義參數的目標
dataset-generation-custom:
	@echo "生成自定義資料集..."
	cd tools/dataset_generators && PYTHONPATH=../.. python generate_training_datasets.py \
		--leaf_file $(LEAF_FILE) \
		--root_file $(ROOT_FILE) \
		--output_dir training_data \
		--k $(K) \
		--sample_size $(SAMPLE_SIZE) \
		--embedding_dim $(EMBEDDING_DIM) \
		--use_clip

train-triplet-custom:
	@echo "訓練自定義 Triplet 模型..."
	cd tools/model_training && PYTHONPATH=../.. python train_triplet_model.py \
		--data_path $(DATA_PATH) \
		--output_dir $(OUTPUT_DIR) \
		--text_embedding_dim $(TEXT_DIM) \
		--image_embedding_dim $(IMAGE_DIM) \
		--hidden_dim $(HIDDEN_DIM) \
		--output_dim $(OUTPUT_DIM) \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--learning_rate $(LR) \
		--device $(DEVICE)

pipeline-custom:
	@echo "執行自定義管道..."
	cd tools/pipelines && PYTHONPATH=../.. python train_and_test_pipeline.py \
		--training_data_dir $(TRAINING_DATA_DIR) \
		--output_dir $(OUTPUT_DIR) \
		--text_embedding_dim $(TEXT_DIM) \
		--image_embedding_dim $(IMAGE_DIM) \
		--hidden_dim $(HIDDEN_DIM) \
		--output_dim $(OUTPUT_DIM) \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--learning_rate $(LR) \
		--similarity_threshold $(THRESHOLD) \
		--device $(DEVICE)
