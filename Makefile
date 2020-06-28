.PHONY: mysql
.PHONY: recorder
.PHONY: server

build:
	docker-compose build

run: build
	docker-compose up -d

stop:
	docker-compose down

logs:
	docker-compose logs --tail=0 --follow

write-logs:
	docker-compose logs --no-color > logfile.log

prune:
	docker image prune

mysql:
	docker exec -it $$(sudo docker ps | grep "mysql" | cut -d' ' -f1) mysql -uroot -p

restart: stop run

recorder:
	docker exec -it $$(sudo docker ps | grep "recorder" | cut -d' ' -f1) bash

server:
	docker exec -it $$(sudo docker ps | grep "server" | cut -d' ' -f1) bash
