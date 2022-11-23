## Тренёв Иван.  
## Группа ML-21.

### Start service in docker:
Build a docker image
~~~
cd online_inference
docker build -t ivantrenev/online_inference:latest .
~~~

Pull a docker image from docker hub
~~~
cd online_inference
docker pull ivantrenev/online_inference:latest
~~~

### Running a container with a docker image:
~~~
docker run --rm -p 9000:9000 ivantrenev/online_inference:latest
~~~

### Docker Optimisation
Ubuntu image (>4000 Mb) -> python:3.8.10 Image (~1500 Mb) -> python:3.8.10-slim (~700 Mb) ->
Use only necessary dependencies, fewer layers and copied files (~600 Mb) 
