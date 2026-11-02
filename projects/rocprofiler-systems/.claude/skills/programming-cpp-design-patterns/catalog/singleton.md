# Singleton

Catalog id: `singleton`

Callers want one object reachable from many places. Pass that object. A request for global access is not a singleton requirement.

## Implement this

```cpp
class logger
{
public:
    void info(std::string_view message);
};

void
run_worker(logger& log)
{
    log.info("worker started");
}
```

The owner creates the logger and passes it. Tests pass a different logger.

## Pattern only if the object is process-lifetime, mutable, and tests can replace the instance

All three must be true. Convenient access from every translation unit is not one of them.

```cpp
class logger
{
public:
    [[nodiscard]] static logger& instance()
    {
        static logger s_value;
        return s_value;
    }

    void info(std::string_view message);

    logger(const logger&)            = delete;
    logger& operator=(const logger&) = delete;

private:
    logger() = default;
};
```
