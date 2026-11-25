# Template Method

Catalog id: `template-method`

The step order is fixed and some steps vary. Pass the varying steps in.

## Implement this

```cpp
template <typename Step>
void
run(Step step2)
{
    step1();
    step2();
    step3();
}
```

## Pattern only if the skeleton must be inherited and steps overridden

```cpp
class abstract_workflow_interface
{
public:
    virtual ~abstract_workflow_interface() = default;

    void run()
    {
        step1();
        step2();
        step3();
    }

protected:
    void         step1() {}
    virtual void step2() = 0;
    void         step3() {}
};
```
