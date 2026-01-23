from locust import HttpUser, task, between 


class LoadTester(HttpUser):
    wait_time = between(1,3)

    @task
    def hit_api(self):
        self.client.get("/api")