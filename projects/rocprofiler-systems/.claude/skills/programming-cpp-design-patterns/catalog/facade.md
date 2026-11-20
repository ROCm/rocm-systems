# Facade

Catalog id: `facade`

A caller makes a long sequence of subsystem calls. Name that sequence once.

## Implement this

```cpp
void
simple_operation(subsystem_a& part_a, subsystem_b& part_b, subsystem_c& part_c)
{
    part_a.op1();
    part_b.op2();
    part_c.op3();
}
```

## Pattern only if many callers need that same surface as an object

```cpp
class facade
{
public:
    void simple_operation()
    {
        m_part_a.op1();
        m_part_b.op2();
        m_part_c.op3();
    }

private:
    subsystem_a m_part_a;
    subsystem_b m_part_b;
    subsystem_c m_part_c;
};
```
