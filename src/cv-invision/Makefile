PY=python
PARENT_PACKAGE=src

download-samples:
	$(PY) -m $(PARENT_PACKAGE).utils.download_samples \
		$(if $(url), --url $(url)) \
		$(if $(output), --output $(output))


ffmpeg-compress-video:
	$(PY) -m $(PARENT_PACKAGE).utils.ffmpeg_video_compress \
		$(if $(input), --input $(input)) \
		$(if $(output), --output $(output)) \
		$(if $(crf), --crf $(crf), --crf 28) \
		$(if $(preset), --preset $(preset), --preset fast)


compress-images:
	$(PY) -m $(PARENT_PACKAGE).utils.image_compress \
		$(if $(input), $(input)) \
		$(if $(size), --size $(size)) \
		$(if $(quality), --quality $(quality))


benchmark:
	$(PY) -m src.model.test.benchmark \
		$(if $(input), --input $(input)) \
		$(if $(mode), --mode $(mode), --mode compare) \
		$(if $(runs), --runs $(runs), --runs 20) \
		$(if $(warmup), --warmup $(warmup), --warmup 3) \
		--profile


run-pipeline:
	$(PY) -m src.pipeline.test.pipeline \
		$(if $(input), --input $(input)) \
		$(if $(output), --output $(output)) \
		$(if $(no_show), --no-show)
