# Facade

Catalog id: `facade`

A caller makes a long sequence of subsystem calls. Name that sequence once.

## Implement this

```cpp
void simple_operation(subsystem_a& a, subsystem_b& b, subsystem_c& c) {
    a.op1();
    b.op2();
    c.op3();
}
```

## Pattern only if many callers need that same surface as an object

```cpp
class facade {
public:
    void simple_operation() {
        a_.op1();
        b_.op2();
        c_.op3();
    }

private:
    subsystem_a a_;
    subsystem_b b_;
    subsystem_c c_;
};
```
