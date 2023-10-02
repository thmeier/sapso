todo:
	ls sapso/*.py | xargs -I % cat % | grep TODO > TODOs.txt

test:
	python3 -i unit_test.py
