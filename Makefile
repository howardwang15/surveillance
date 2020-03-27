build:
	docker-compose build

run: build
	docker-compose up

stop:
	docker-compose down


prune:
	docker image prune
