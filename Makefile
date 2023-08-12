pre:
	python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
	mkdir -p thirdparty2
	git clone https://github.com/open-mmlab/mmdetection.git thirdparty2/mmdetection
	cd thirdparty2/mmdetection && python -m pip install -e .
install:
	make pre
	python -m pip install -e .
clean:
	rm -rf thirdparty2
	rm -r ssod.egg-info
