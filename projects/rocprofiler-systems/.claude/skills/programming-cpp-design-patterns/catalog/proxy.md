# Proxy

Catalog id: `proxy`

Access needs a check, a delay, or a stand-in. Put the check at the call when the caller already has the real object.

## Implement this

```cpp
void request(real_subject& subject) {
    subject.request();
}
```

## Pattern only if the stand-in must have the subject's interface

```cpp
class proxy : public subject {
public:
    void request() override {
        if (!real_) real_ = std::make_unique<real_subject>();
        real_->request();
    }

private:
    std::unique_ptr<real_subject> real_;
};
```

Use the same shape for an access check or a cache in front of `real_->request()`.
