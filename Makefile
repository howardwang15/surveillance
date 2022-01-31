.PHONY: mysql
.PHONY: recorder
.PHONY: server

build:
	docker-compose build

run: build
	docker-compose up -d

dev: build
	docker-compose up

stop:
	docker-compose down

logs:
	docker-compose logs --tail=0 --follow

write-logs:
	docker-compose logs --no-color > logfile.log

prune:
	docker image prune

mysql:
	docker exec -it $$(docker ps | grep "mysql" | cut -d' ' -f1) mysql -uroot -p

restart: stop run

recorder:
	docker exec -it $$(docker ps | grep "recorder" | cut -d' ' -f1) bash

server:
	docker exec -it $$(docker ps | grep "server" | cut -d' ' -f1) bash
