build:
	docker-compose build

run: build
	docker-compose up

stop:
	docker-compose down


prune:
	docker image prune

mysql:
	docker exec -it $$(sudo docker ps | grep "mysql" | cut -d' ' -f1) mysql -uroot -p

server:
	docker exec -it $$(sudo docker ps | grep "server" | cut -d' ' -f1) bash
