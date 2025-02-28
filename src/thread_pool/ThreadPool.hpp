//
// Created by xinyi on 24-12-8.
//

#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>

#define MAX_THREAD_NUM 8

class ThreadPool {
public:
    // 构造函数，指定线程池中线程的数量
    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        // 获取锁，检查任务队列
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });

                        if (this->stop && this->tasks.empty()) {
                            return;
                        }

                        // 获取任务并从队列中移除
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    // 执行任务
                    task();
                }
            });
        }
    }

    // 向线程池添加任务，任务返回值为T
    template <typename F, typename... Args>
    std::future<typename std::result_of<F(Args...)>::type> enqueue(F&& f, Args&&... args) {
        using return_type = typename std::result_of<F(Args...)>::type;

        // 通过包装函数，创建一个任务
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.push([task] { (*task)(); });
        }
        condition.notify_one();

        return task->get_future(); // 返回future，便于获取任务结果
    }


    // 停止线程池，等待所有线程完成任务
    void stopPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all(); // 通知所有线程停止

        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join(); // 等待每个线程完成任务
            }
        }
    }

    ~ThreadPool() {
        if (!stop) {
            stopPool();
        }
    }

private:
    std::vector<std::thread> workers;               // 工作线程
    std::queue<std::function<void()>> tasks;        // 任务队列
    std::mutex queueMutex;                          // 任务队列锁
    std::condition_variable condition;              // 条件变量
    std::atomic_bool stop;                          // 标志线程池是否停止
};



#endif //THREADPOOL_HPP
