todo:
	ls sapso/*.py | xargs -I % cat % | grep TODO > TODOs.txt

test:
	python3 unit_test.py
