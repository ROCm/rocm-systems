---
name: refactoring-techniques
description: Catalog of refactoring techniques from refactoring.guru - how to apply Extract Method, Move Field, Replace Conditional, and 60+ other refactorings
---

# Refactoring Techniques Catalog

Comprehensive reference of refactoring techniques for transforming code while preserving behavior. Based on [Refactoring.Guru Techniques](https://refactoring.guru/refactoring/techniques).

<IMPORTANT>
This skill is a **reference catalog** for applying refactorings. It is used by:
- `planning-refactor` when implementing improvements
- `code-smells` to suggest fixes for detected smells
- Standalone refactoring assistance

**Cross-Reference:** See `code-smells` skill for detecting WHAT to fix. This skill shows HOW to fix it.
</IMPORTANT>

## When to Use

| Context | Trigger |
| --------- | --------- |
| **Planning Refactor** | Need specific technique to fix identified smell |
| **Code Review** | Suggesting improvements during PR review |
| **Learning** | User asks how to apply a specific refactoring |
| **Quick Fix** | User wants to apply a refactoring immediately |

## Quick Reference: Smell to Refactoring

| Code Smell | Recommended Refactorings |
| ------------ | ------------------------- |
| Long Method | Extract Method, Replace Temp with Query, Decompose Conditional |
| Large Class | Extract Class, Extract Subclass, Extract Interface |
| Long Parameter List | Introduce Parameter Object, Preserve Whole Object |
| Duplicate Code | Extract Method, Pull Up Method, Form Template Method |
| Feature Envy | Move Method, Extract Method |
| Switch Statements | Replace Conditional with Polymorphism, Replace Type Code with Subclasses |
| Data Clumps | Extract Class, Introduce Parameter Object |
| Primitive Obsession | Replace Data Value with Object, Replace Type Code with Class |
| Shotgun Surgery | Move Method, Move Field, Inline Class |
| Divergent Change | Extract Class |
| Message Chains | Hide Delegate |
| Middle Man | Remove Middle Man |
| Refused Bequest | Replace Inheritance with Delegation |

---

## Category 1: Composing Methods

Techniques for correctly structuring methods. Long methods are often the root of many problems.

### Extract Method

| Aspect | Details |
| -------- | --------- |
| **Impact** | High - Most frequently used refactoring |
| **Risk** | Low |
| **Fixes** | Long Method, Duplicate Code, Comments |

**Problem:** Code fragment that can be logically grouped together.

**Solution:** Create a new method with a descriptive name and move the fragment there.

**Steps:**

1. Create new method named after WHAT it does (not HOW)
2. Copy the code fragment to the new method
3. Look for local variables used only within the fragment - make them local to new method
4. Pass external variables as parameters
5. If extracted code modifies a variable needed later, return it
6. Replace original fragment with method call

**C++ Example:**

```cpp
// BEFORE
void print_invoice(const Invoice& inv) {
    // Print header
    std::cout << "================\n";
    std::cout << "INVOICE #" << inv.number << "\n";
    std::cout << "Date: " << inv.date << "\n";
    std::cout << "================\n";

    // Print items
    for (const auto& item : inv.items) {
        std::cout << item.name << ": $" << item.price << "\n";
    }

    // Calculate and print total
    double total = 0;
    for (const auto& item : inv.items) {
        total += item.price;
    }
    std::cout << "Total: $" << total << "\n";
}

// AFTER
void print_header(const Invoice& inv) {
    std::cout << "================\n";
    std::cout << "INVOICE #" << inv.number << "\n";
    std::cout << "Date: " << inv.date << "\n";
    std::cout << "================\n";
}

void print_items(const std::vector<Item>& items) {
    for (const auto& item : items) {
        std::cout << item.name << ": $" << item.price << "\n";
    }
}

double calculate_total(const std::vector<Item>& items) {
    return std::accumulate(items.begin(), items.end(), 0.0,
        [](double sum, const Item& item) { return sum + item.price; });
}

void print_invoice(const Invoice& inv) {
    print_header(inv);
    print_items(inv.items);
    std::cout << "Total: $" << calculate_total(inv.items) << "\n";
}
```

**Python Example:**

```python
# BEFORE
def print_invoice(inv):
    # Print header
    print("================")
    print(f"INVOICE #{inv.number}")
    print(f"Date: {inv.date}")
    print("================")

    # Print items and calculate total
    total = 0
    for item in inv.items:
        print(f"{item.name}: ${item.price}")
        total += item.price
    print(f"Total: ${total}")

# AFTER
def print_header(inv):
    print("================")
    print(f"INVOICE #{inv.number}")
    print(f"Date: {inv.date}")
    print("================")

def print_items(items):
    for item in items:
        print(f"{item.name}: ${item.price}")

def calculate_total(items):
    return sum(item.price for item in items)

def print_invoice(inv):
    print_header(inv)
    print_items(inv.items)
    print(f"Total: ${calculate_total(inv.items)}")
```

---

### Inline Method

| Aspect | Details |
| -------- | --------- |
| **Impact** | Low |
| **Risk** | Low |
| **Fixes** | Speculative Generality, excessive delegation |

**Problem:** Method body is more obvious than the method itself.

**Solution:** Replace method calls with the method content and delete the method.

**Steps:**

1. Verify method is not overridden in subclasses
2. Find all calls to the method
3. Replace each call with the method body
4. Delete the method

```cpp
// BEFORE
int get_rating() {
    return more_than_five_late_deliveries() ? 2 : 1;
}
bool more_than_five_late_deliveries() {
    return late_deliveries_ > 5;
}

// AFTER
int get_rating() {
    return late_deliveries_ > 5 ? 2 : 1;
}
```

---

### Extract Variable

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Complex expressions, Comments |

**Problem:** Hard-to-understand expression.

**Solution:** Place the expression result in a self-explanatory variable.

**Steps:**

1. Insert a new variable and assign the expression to it
2. Replace the original expression with the variable
3. Repeat for other occurrences of the same expression

```cpp
// BEFORE
if (platform.find("MAC") != std::string::npos &&
    browser.find("IE") != std::string::npos &&
    was_initialized() && resize > 0) {
    // ...
}

// AFTER
bool is_mac = platform.find("MAC") != std::string::npos;
bool is_ie = browser.find("IE") != std::string::npos;
bool was_resized = resize > 0;

if (is_mac && is_ie && was_initialized() && was_resized) {
    // ...
}
```

---

### Replace Temp with Query

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Long Method (preparation for Extract Method) |

**Problem:** Temporary variable storing an expression result for later use.

**Solution:** Move the expression to a new method and call the method instead of using the variable.

**Steps:**

1. Ensure the variable is only assigned once
2. Extract the expression into a new method
3. Replace all uses of the variable with method calls
4. Remove the variable declaration

```cpp
// BEFORE
double calculate_total() {
    double base_price = quantity_ * item_price_;
    if (base_price > 1000) {
        return base_price * 0.95;
    }
    return base_price * 0.98;
}

// AFTER
double base_price() const {
    return quantity_ * item_price_;
}

double calculate_total() {
    if (base_price() > 1000) {
        return base_price() * 0.95;
    }
    return base_price() * 0.98;
}
```

---

### Split Temporary Variable

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Variable reuse, unclear intent |

**Problem:** Local variable used for multiple unrelated purposes.

**Solution:** Create separate variables for each purpose.

```cpp
// BEFORE
double temp = 2 * (height_ + width_);
std::cout << "Perimeter: " << temp << "\n";
temp = height_ * width_;
std::cout << "Area: " << temp << "\n";

// AFTER
double perimeter = 2 * (height_ + width_);
std::cout << "Perimeter: " << perimeter << "\n";
double area = height_ * width_;
std::cout << "Area: " << area << "\n";
```

---

### Remove Assignments to Parameters

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Confusing parameter modification |

**Problem:** Value assigned to a parameter inside method body.

**Solution:** Use a local variable instead of the parameter.

```cpp
// BEFORE
int discount(int input_val, int quantity) {
    if (quantity > 50) {
        input_val -= 2;  // Modifying parameter!
    }
    return input_val;
}

// AFTER
int discount(int input_val, int quantity) {
    int result = input_val;
    if (quantity > 50) {
        result -= 2;
    }
    return result;
}
```

---

### Replace Method with Method Object

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Long Method with intertwined local variables |

**Problem:** Long method where local variables prevent extraction.

**Solution:** Transform the method into a separate class where locals become fields.

**Steps:**

1. Create a new class named after the method
2. Create a private field for the original object and each local variable/parameter
3. Create a constructor that initializes all fields
4. Copy the method body to a `compute()` method in the new class
5. Replace original method with creation of method object and call to `compute()`

```cpp
// BEFORE
class Order {
public:
    double calculate_price() {
        double primary_base = // complex calculation
        double secondary = // uses primary_base
        double tertiary = // uses secondary
        // 100 more lines using all these variables...
        return result;
    }
};

// AFTER
class PriceCalculator {
    Order& order_;
    double primary_base_;
    double secondary_;
    double tertiary_;

public:
    PriceCalculator(Order& order) : order_(order) {}

    double compute() {
        primary_base_ = // complex calculation
        secondary_ = // uses primary_base_
        tertiary_ = // uses secondary_
        // Can now Extract Method freely!
        return calculate_final();
    }

private:
    double calculate_final() { /*...*/ }
};

class Order {
public:
    double calculate_price() {
        return PriceCalculator(*this).compute();
    }
};
```

---

### Substitute Algorithm

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | High |
| **Fixes** | Inefficient or unclear algorithm |

**Problem:** Want to replace an algorithm with a better one.

**Solution:** Replace the method body with the new algorithm.

**Steps:**

1. Simplify the existing algorithm first
2. Create tests covering all edge cases
3. Write the new algorithm
4. Run tests to verify behavior is identical
5. If tests fail, use old algorithm for debugging

```cpp
// BEFORE
std::string find_person(const std::vector<std::string>& people) {
    for (const auto& person : people) {
        if (person == "Don") return "Don";
    }
    for (const auto& person : people) {
        if (person == "John") return "John";
    }
    for (const auto& person : people) {
        if (person == "Kent") return "Kent";
    }
    return "";
}

// AFTER
std::string find_person(const std::vector<std::string>& people) {
    static const std::vector<std::string> candidates = {"Don", "John", "Kent"};
    for (const auto& candidate : candidates) {
        if (std::find(people.begin(), people.end(), candidate) != people.end()) {
            return candidate;
        }
    }
    return "";
}
```

---

## Category 2: Moving Features Between Objects

Techniques for distributing functionality optimally between classes.

### Move Method

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Feature Envy, Shotgun Surgery |

**Problem:** Method is used more by another class than its own class.

**Solution:** Move the method to the class that uses it most.

**Steps:**

1. Examine all features used by the method in its current class
2. Check if method is declared in parent/child classes (may need polymorphism)
3. Declare the method in the target class
4. Copy the method body, adapting references to the new context
5. Convert original method to delegate to the new location (or remove it)
6. Update all callers

```cpp
// BEFORE
class Account {
    AccountType type_;
public:
    double overdraft_charge() {
        if (type_.is_premium()) {
            double result = 10;
            if (days_overdrawn_ > 7) {
                result += (days_overdrawn_ - 7) * 0.85;
            }
            return result;
        }
        return days_overdrawn_ * 1.75;
    }
};

// AFTER - Method moved to AccountType where it belongs
class AccountType {
public:
    double overdraft_charge(int days_overdrawn) {
        if (is_premium()) {
            double result = 10;
            if (days_overdrawn > 7) {
                result += (days_overdrawn - 7) * 0.85;
            }
            return result;
        }
        return days_overdrawn * 1.75;
    }
};

class Account {
public:
    double overdraft_charge() {
        return type_.overdraft_charge(days_overdrawn_);
    }
};
```

---

### Move Field

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Feature Envy, Data Clumps |

**Problem:** Field is used more by another class than its own class.

**Solution:** Create a field in the target class and redirect all users.

**Steps:**

1. If field is public, encapsulate it first
2. Create field and accessors in target class
3. Determine how to reference the target from the source
4. Update all references to use the new location
5. Remove the field from the original class

```cpp
// BEFORE
class Account {
    AccountType type_;
    double interest_rate_;
};

// AFTER - interest_rate moved to AccountType
class AccountType {
    double interest_rate_;
public:
    double interest_rate() const { return interest_rate_; }
    void set_interest_rate(double rate) { interest_rate_ = rate; }
};

class Account {
    AccountType type_;
public:
    double interest_rate() const { return type_.interest_rate(); }
};
```

---

### Extract Class

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Large Class, Divergent Change, Data Clumps |

**Problem:** One class does the work of two.

**Solution:** Create a new class and move relevant fields and methods.

**Steps:**

1. Decide what to split out
2. Create a new class for the extracted responsibility
3. Create a link from the old class to the new one
4. Use Move Field for each field to move
5. Use Move Method for each method to move
6. Review and reduce interfaces of both classes
7. Decide on new class visibility (public or encapsulated)

```cpp
// BEFORE
class Person {
    std::string name_;
    std::string office_area_code_;
    std::string office_number_;
public:
    std::string telephone_number() {
        return "(" + office_area_code_ + ") " + office_number_;
    }
};

// AFTER
class TelephoneNumber {
    std::string area_code_;
    std::string number_;
public:
    std::string format() const {
        return "(" + area_code_ + ") " + number_;
    }
    // getters/setters...
};

class Person {
    std::string name_;
    TelephoneNumber office_telephone_;
public:
    std::string telephone_number() {
        return office_telephone_.format();
    }
};
```

---

### Inline Class

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Lazy Class, Speculative Generality |

**Problem:** Class does almost nothing and has no planned responsibilities.

**Solution:** Move all features to another class and delete the empty class.

**Steps:**

1. Make all public members of the source class public in the target
2. Change all references to point to the target class
3. Move methods and fields one by one to the target
4. Delete the empty source class

```cpp
// BEFORE - TelephoneNumber is too thin
class TelephoneNumber {
    std::string number_;
public:
    std::string number() const { return number_; }
};

class Person {
    TelephoneNumber phone_;
public:
    std::string phone_number() { return phone_.number(); }
};

// AFTER
class Person {
    std::string phone_number_;
public:
    std::string phone_number() const { return phone_number_; }
};
```

---

### Hide Delegate

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Message Chains, tight coupling |

**Problem:** Client calls object B from object A, then calls methods on B.

**Solution:** Create a method in A that delegates to B, hiding B from the client.

```cpp
// BEFORE - Client knows about Department
class Person {
    Department* department_;
public:
    Department* department() { return department_; }
};

// Client code:
manager = john.department()->manager();

// AFTER - Department hidden
class Person {
    Department* department_;
public:
    Person* manager() { return department_->manager(); }
};

// Client code:
manager = john.manager();
```

---

### Remove Middle Man

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Middle Man |

**Problem:** Class has too many methods that simply delegate.

**Solution:** Delete delegation methods and have clients call the delegate directly.

```cpp
// BEFORE - Person delegates everything
class Person {
    Department* department_;
public:
    Person* manager() { return department_->manager(); }
    std::string budget() { return department_->budget(); }
    std::string team_size() { return department_->team_size(); }
    // ... 10 more delegation methods
};

// AFTER - Expose delegate for complex queries
class Person {
    Department* department_;
public:
    Department* department() { return department_; }
};

// Client handles department directly when needed
auto budget = john.department()->budget();
```

---

### Introduce Foreign Method

| Aspect | Details |
| -------- | --------- |
| **Impact** | Low |
| **Risk** | Low |
| **Fixes** | Incomplete Library Class (1-2 methods) |

**Problem:** Utility class lacks a method you need, and you cannot modify it.

**Solution:** Create the method in your client class, taking the utility object as a parameter.

```cpp
// Can't modify Date class, need next_day functionality
class Report {
    // Foreign method - ideally should be in Date class
    static Date next_day(const Date& date) {
        return Date(date.year(), date.month(), date.day() + 1);
    }

    void generate() {
        Date start = get_start_date();
        Date end = next_day(start);  // Using foreign method
        // ...
    }
};
```

---

### Introduce Local Extension

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Medium |
| **Fixes** | Incomplete Library Class (many methods) |

**Problem:** Utility class lacks several methods you need.

**Solution:** Create a new class (subclass or wrapper) containing the methods.

```cpp
// Subclass approach
class ExtendedDate : public Date {
public:
    using Date::Date;  // Inherit constructors

    Date next_day() const {
        return Date(year(), month(), day() + 1);
    }

    bool is_weekend() const {
        int dow = day_of_week();
        return dow == 0 || dow == 6;
    }
};

// Wrapper approach (when inheritance not possible)
class DateWrapper {
    Date date_;
public:
    explicit DateWrapper(const Date& d) : date_(d) {}

    // Delegate existing methods
    int year() const { return date_.year(); }
    int month() const { return date_.month(); }
    int day() const { return date_.day(); }

    // Add new methods
    Date next_day() const {
        return Date(year(), month(), day() + 1);
    }
};
```

---

## Category 3: Organizing Data

Techniques for improving data handling and class associations.

### Self Encapsulate Field

| Aspect | Details |
| -------- | --------- |
| **Impact** | Low |
| **Risk** | Low |
| **Fixes** | Direct field access issues |

**Problem:** Direct access to private fields inside the class.

**Solution:** Create getter/setter and use them internally.

```cpp
// BEFORE
class IntRange {
    int low_, high_;
public:
    bool includes(int arg) {
        return arg >= low_ && arg <= high_;  // Direct access
    }
};

// AFTER
class IntRange {
    int low_, high_;
public:
    int low() const { return low_; }
    int high() const { return high_; }

    bool includes(int arg) {
        return arg >= low() && arg <= high();  // Via getters
    }
};
```

---

### Replace Data Value with Object

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Primitive Obsession, Data Clumps |

**Problem:** Data field has behavior or associated data.

**Solution:** Turn the field into a class.

```cpp
// BEFORE
class Order {
    std::string customer_name_;
    std::string customer_email_;
    std::string customer_phone_;
};

// AFTER
class Customer {
    std::string name_;
    std::string email_;
    std::string phone_;
public:
    // Behavior can live here
    bool is_valid_email() const;
    std::string formatted_phone() const;
};

class Order {
    Customer customer_;
};
```

---

### Replace Magic Number with Symbolic Constant

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Magic Numbers |

**Problem:** Code uses a number with special meaning.

**Solution:** Replace with a named constant.

```cpp
// BEFORE
double potential_energy(double mass, double height) {
    return mass * 9.81 * height;  // Magic number!
}

// AFTER
constexpr double GRAVITATIONAL_ACCELERATION = 9.81;  // m/s^2

double potential_energy(double mass, double height) {
    return mass * GRAVITATIONAL_ACCELERATION * height;
}
```

---

### Encapsulate Field

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Public fields, Data Class |

**Problem:** You have a public field.

**Solution:** Make it private and provide accessors.

```cpp
// BEFORE
class Person {
public:
    std::string name;  // Public field!
};

// AFTER
class Person {
    std::string name_;
public:
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }
};
```

---

### Encapsulate Collection

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Collection exposure, broken encapsulation |

**Problem:** Method returns a collection that can be modified externally.

**Solution:** Return read-only view and provide add/remove methods.

```cpp
// BEFORE
class Person {
    std::vector<Course> courses_;
public:
    std::vector<Course>& courses() { return courses_; }  // Dangerous!
};

// Client can: person.courses().clear();  // Bypass Person!

// AFTER
class Person {
    std::vector<Course> courses_;
public:
    const std::vector<Course>& courses() const { return courses_; }

    void add_course(const Course& c) {
        courses_.push_back(c);
    }

    void remove_course(const Course& c) {
        courses_.erase(
            std::remove(courses_.begin(), courses_.end(), c),
            courses_.end());
    }
};
```

---

### Replace Type Code with Class

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Primitive Obsession (type codes) |

**Problem:** Class has a field with type code that doesn't affect behavior.

**Solution:** Replace with a new class.

```cpp
// BEFORE
class Person {
    int blood_type_;  // 0=O, 1=A, 2=B, 3=AB
public:
    static constexpr int O = 0;
    static constexpr int A = 1;
    static constexpr int B = 2;
    static constexpr int AB = 3;
};

// AFTER
class BloodType {
    int code_;
    BloodType(int code) : code_(code) {}
public:
    static const BloodType O;
    static const BloodType A;
    static const BloodType B;
    static const BloodType AB;

    bool operator==(const BloodType& other) const {
        return code_ == other.code_;
    }
};

const BloodType BloodType::O{0};
const BloodType BloodType::A{1};
const BloodType BloodType::B{2};
const BloodType BloodType::AB{3};

class Person {
    BloodType blood_type_;
};
```

---

### Replace Type Code with Subclasses

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | High |
| **Fixes** | Switch Statements, Primitive Obsession |

**Problem:** Type code that directly affects behavior via conditionals.

**Solution:** Create subclasses for each type code value.

```cpp
// BEFORE
class Employee {
    int type_;  // 0=Engineer, 1=Manager, 2=Salesman
public:
    static constexpr int ENGINEER = 0;
    static constexpr int MANAGER = 1;
    static constexpr int SALESMAN = 2;

    int bonus() {
        switch (type_) {
            case ENGINEER: return 100;
            case MANAGER: return 500;
            case SALESMAN: return 200;
        }
        return 0;
    }
};

// AFTER
class Employee {
public:
    virtual int bonus() = 0;
};

class Engineer : public Employee {
public:
    int bonus() override { return 100; }
};

class Manager : public Employee {
public:
    int bonus() override { return 500; }
};

class Salesman : public Employee {
public:
    int bonus() override { return 200; }
};
```

---

### Replace Type Code with State/Strategy

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | High |
| **Fixes** | Switch Statements when subclassing is impossible |

**Problem:** Type code affects behavior but you cannot use subclasses (type changes at runtime).

**Solution:** Replace type code with a state object.

```cpp
// BEFORE
class Employee {
    int type_;
    void set_type(int type) { type_ = type; }  // Type can change!
public:
    int pay_amount() {
        switch (type_) {
            case ENGINEER: return monthly_salary_;
            case SALESMAN: return monthly_salary_ + commission_;
            case MANAGER: return monthly_salary_ + bonus_;
        }
    }
};

// AFTER
class EmployeeType {
public:
    virtual int pay_amount(const Employee& emp) = 0;
};

class Engineer : public EmployeeType {
public:
    int pay_amount(const Employee& emp) override {
        return emp.monthly_salary();
    }
};

class Employee {
    std::unique_ptr<EmployeeType> type_;
public:
    void set_type(std::unique_ptr<EmployeeType> t) {
        type_ = std::move(t);  // Can change type at runtime
    }
    int pay_amount() {
        return type_->pay_amount(*this);
    }
};
```

---

## Category 4: Simplifying Conditional Expressions

Techniques for managing conditional complexity.

### Decompose Conditional

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Low |
| **Fixes** | Long Method, Complex conditionals |

**Problem:** Complex conditional (if-then-else or switch).

**Solution:** Extract condition, then-block, and else-block into separate methods.

```cpp
// BEFORE
double calculate_charge(const Date& date, int quantity) {
    if (date < SUMMER_START || date > SUMMER_END) {
        return quantity * winter_rate_ + winter_service_charge_;
    } else {
        return quantity * summer_rate_;
    }
}

// AFTER
bool is_summer(const Date& date) {
    return date >= SUMMER_START && date <= SUMMER_END;
}

double summer_charge(int quantity) {
    return quantity * summer_rate_;
}

double winter_charge(int quantity) {
    return quantity * winter_rate_ + winter_service_charge_;
}

double calculate_charge(const Date& date, int quantity) {
    if (is_summer(date)) {
        return summer_charge(quantity);
    }
    return winter_charge(quantity);
}
```

---

### Consolidate Conditional Expression

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Duplicate conditionals |

**Problem:** Multiple conditionals leading to the same result.

**Solution:** Combine into a single expression and extract to a method.

```cpp
// BEFORE
double disability_amount() {
    if (seniority_ < 2) return 0;
    if (months_disabled_ > 12) return 0;
    if (is_part_time_) return 0;
    // Calculate disability
    return calculate_disability();
}

// AFTER
bool is_not_eligible_for_disability() {
    return seniority_ < 2
        || months_disabled_ > 12
        || is_part_time_;
}

double disability_amount() {
    if (is_not_eligible_for_disability()) return 0;
    return calculate_disability();
}
```

---

### Consolidate Duplicate Conditional Fragments

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Duplicate Code in branches |

**Problem:** Same code in all branches of a conditional.

**Solution:** Move it outside the conditional.

```cpp
// BEFORE
if (is_special_deal()) {
    total = price * 0.95;
    send_notification();  // Duplicated!
} else {
    total = price * 0.98;
    send_notification();  // Duplicated!
}

// AFTER
if (is_special_deal()) {
    total = price * 0.95;
} else {
    total = price * 0.98;
}
send_notification();  // Moved outside
```

---

### Replace Nested Conditional with Guard Clauses

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Low |
| **Fixes** | Deep nesting, unclear control flow |

**Problem:** Nested conditionals make normal flow hard to see.

**Solution:** Use guard clauses for special cases, returning early.

```cpp
// BEFORE
double get_pay_amount() {
    double result;
    if (is_dead_) {
        result = dead_amount();
    } else {
        if (is_separated_) {
            result = separated_amount();
        } else {
            if (is_retired_) {
                result = retired_amount();
            } else {
                result = normal_pay_amount();
            }
        }
    }
    return result;
}

// AFTER
double get_pay_amount() {
    if (is_dead_) return dead_amount();
    if (is_separated_) return separated_amount();
    if (is_retired_) return retired_amount();
    return normal_pay_amount();
}
```

---

### Replace Conditional with Polymorphism

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | High |
| **Fixes** | Switch Statements, Type-based conditionals |

**Problem:** Conditional that varies behavior based on object type.

**Solution:** Create subclasses and override the method.

**Steps:**

1. If conditional is part of larger method, use Extract Method first
2. If conditional is in a class that has other responsibilities, Extract Class first
3. Create subclasses for each conditional branch
4. Create abstract method in superclass
5. Override method in each subclass with branch logic
6. Delete branches from original conditional
7. Delete the conditional; make method abstract

```cpp
// BEFORE
class Bird {
    std::string type_;
public:
    double get_speed() {
        switch (type_) {
            case "European": return get_base_speed();
            case "African":
                return get_base_speed() - load_factor_ * num_coconuts_;
            case "NorwegianBlue":
                return is_nailed_ ? 0 : get_base_speed(voltage_);
        }
    }
};

// AFTER
class Bird {
public:
    virtual double get_speed() = 0;
};

class European : public Bird {
public:
    double get_speed() override {
        return get_base_speed();
    }
};

class African : public Bird {
public:
    double get_speed() override {
        return get_base_speed() - load_factor_ * num_coconuts_;
    }
};

class NorwegianBlue : public Bird {
public:
    double get_speed() override {
        return is_nailed_ ? 0 : get_base_speed(voltage_);
    }
};
```

---

### Introduce Null Object

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Repeated null checks |

**Problem:** Many null checks throughout the code.

**Solution:** Return a null object that exhibits default behavior instead of null.

```cpp
// BEFORE
class Customer {
    BillingPlan* plan_;
public:
    BillingPlan* plan() { return plan_; }  // May return nullptr
};

// Usage scattered with null checks:
if (customer != nullptr) {
    plan = customer->plan();
} else {
    plan = BillingPlan::basic();
}

// AFTER
class NullCustomer : public Customer {
public:
    bool is_null() override { return true; }
    BillingPlan* plan() override { return &BillingPlan::basic(); }
};

class Customer {
public:
    virtual bool is_null() { return false; }
    virtual BillingPlan* plan() { return plan_; }
};

// Usage - no null checks needed:
plan = customer->plan();
```

---

### Introduce Assertion

| Aspect | Details |
| -------- | --------- |
| **Impact** | Low |
| **Risk** | Low |
| **Fixes** | Implicit assumptions |

**Problem:** Code assumes certain conditions are true.

**Solution:** Make assumptions explicit with assertions.

```cpp
// BEFORE
double get_expense_limit() {
    // Assumes either expense_limit or primary_project is set
    return (expense_limit_ != NULL_EXPENSE)
        ? expense_limit_
        : primary_project_->member_expense_limit();
}

// AFTER
double get_expense_limit() {
    assert(expense_limit_ != NULL_EXPENSE || primary_project_ != nullptr);
    return (expense_limit_ != NULL_EXPENSE)
        ? expense_limit_
        : primary_project_->member_expense_limit();
}
```

---

## Category 5: Simplifying Method Calls

Techniques for making method interfaces cleaner.

### Rename Method

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Low |
| **Fixes** | Unclear intent, Comments |

**Problem:** Method name doesn't explain what it does.

**Solution:** Rename it.

```cpp
// BEFORE
std::string get_tlp() { return telephone_; }

// AFTER
std::string get_telephone_number() { return telephone_; }
```

---

### Add Parameter / Remove Parameter

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Medium |
| **Fixes** | Missing/unused data |

**Add Parameter:** Method needs data it doesn't have.
**Remove Parameter:** Parameter is unused.

```cpp
// Add Parameter
// BEFORE: Date fetched from global state
Date get_date() { return Date::today(); }

// AFTER: Date passed explicitly
Date get_date(const Date& base_date) { return base_date; }

// Remove Parameter
// BEFORE: Unused parameter
void set_value(int value, int unused) { value_ = value; }

// AFTER
void set_value(int value) { value_ = value; }
```

---

### Separate Query from Modifier

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Side effects, hard to test |

**Problem:** Method returns value AND changes object state.

**Solution:** Split into two methods: one returns value, one modifies state.

```cpp
// BEFORE
int get_total_and_clear() {
    int result = calculate_total();
    clear_items();  // Side effect!
    return result;
}

// AFTER
int get_total() const {
    return calculate_total();  // Query - no side effects
}

void clear() {
    clear_items();  // Modifier - no return value
}
```

---

### Parameterize Method

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Similar methods, Duplicate Code |

**Problem:** Multiple methods do similar things with different values.

**Solution:** Create one method with a parameter.

```cpp
// BEFORE
void five_percent_raise() { salary_ *= 1.05; }
void ten_percent_raise() { salary_ *= 1.10; }
void fifteen_percent_raise() { salary_ *= 1.15; }

// AFTER
void raise(double factor) { salary_ *= (1 + factor); }

// Usage: raise(0.05), raise(0.10), raise(0.15)
```

---

### Introduce Parameter Object

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Long Parameter List, Data Clumps |

**Problem:** Methods have repeating parameter groups.

**Solution:** Replace with an object containing those parameters.

```cpp
// BEFORE
void amount_invoiced(Date start, Date end);
void amount_received(Date start, Date end);
void amount_overdue(Date start, Date end);

// AFTER
class DateRange {
    Date start_, end_;
public:
    DateRange(Date start, Date end) : start_(start), end_(end) {}
    Date start() const { return start_; }
    Date end() const { return end_; }
    bool includes(Date d) const { return d >= start_ && d <= end_; }
};

void amount_invoiced(DateRange range);
void amount_received(DateRange range);
void amount_overdue(DateRange range);
```

---

### Preserve Whole Object

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Long Parameter List, Feature Envy |

**Problem:** Getting several values from an object then passing them as parameters.

**Solution:** Pass the whole object instead.

```cpp
// BEFORE
int low = days_temp_range.low();
int high = days_temp_range.high();
bool within_plan = plan.within_range(low, high);

// AFTER
bool within_plan = plan.within_range(days_temp_range);
```

---

### Replace Constructor with Factory Method

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Complex construction, Type-based creation |

**Problem:** Constructor does more than setting fields, or you need type-based creation.

**Solution:** Replace with factory method.

```cpp
// BEFORE
class Employee {
    int type_;
public:
    Employee(int type) : type_(type) {}
};

// AFTER
class Employee {
protected:
    Employee() = default;
public:
    static std::unique_ptr<Employee> create(int type) {
        switch (type) {
            case ENGINEER: return std::make_unique<Engineer>();
            case MANAGER: return std::make_unique<Manager>();
            case SALESMAN: return std::make_unique<Salesman>();
        }
        throw std::invalid_argument("Invalid employee type");
    }
};
```

---

### Replace Error Code with Exception

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Error handling clutter |

**Problem:** Method returns special value to indicate error.

**Solution:** Throw an exception instead.

```cpp
// BEFORE
int withdraw(int amount) {
    if (amount > balance_) {
        return -1;  // Error code
    }
    balance_ -= amount;
    return 0;  // Success
}

// Caller must check: if (account.withdraw(100) == -1) { ... }

// AFTER
void withdraw(int amount) {
    if (amount > balance_) {
        throw std::runtime_error("Insufficient funds");
    }
    balance_ -= amount;
}

// Caller: try { account.withdraw(100); } catch (...) { ... }
```

---

### Replace Exception with Test

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Exception misuse |

**Problem:** Exception thrown where a condition check would suffice.

**Solution:** Replace exception with a condition test.

```cpp
// BEFORE
double value_for_period(int period) {
    try {
        return values_.at(period);
    } catch (std::out_of_range&) {
        return 0;
    }
}

// AFTER
double value_for_period(int period) {
    if (period < 0 || period >= values_.size()) {
        return 0;
    }
    return values_[period];
}
```

---

## Category 6: Dealing with Generalization

Techniques for managing inheritance hierarchies.

### Pull Up Field / Pull Up Method

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Duplicate Code in subclasses |

**Problem:** Subclasses have the same field or method.

**Solution:** Move to the superclass.

```cpp
// BEFORE
class Salesman : public Employee {
    std::string name_;  // Duplicated!
};
class Engineer : public Employee {
    std::string name_;  // Duplicated!
};

// AFTER
class Employee {
protected:
    std::string name_;  // Moved to parent
};
class Salesman : public Employee {};
class Engineer : public Employee {};
```

---

### Pull Up Constructor Body

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Duplicate constructor code |

**Problem:** Subclass constructors have mostly identical code.

**Solution:** Create superclass constructor and call from subclasses.

```cpp
// BEFORE
class Manager : public Employee {
public:
    Manager(std::string name, std::string id, int grade) {
        name_ = name;    // Duplicated
        id_ = id;        // Duplicated
        grade_ = grade;
    }
};
class Salesman : public Employee {
public:
    Salesman(std::string name, std::string id, int quota) {
        name_ = name;    // Duplicated
        id_ = id;        // Duplicated
        quota_ = quota;
    }
};

// AFTER
class Employee {
protected:
    Employee(std::string name, std::string id)
        : name_(name), id_(id) {}
};

class Manager : public Employee {
public:
    Manager(std::string name, std::string id, int grade)
        : Employee(name, id), grade_(grade) {}
};
```

---

### Push Down Method / Push Down Field

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Superclass has irrelevant members |

**Problem:** Behavior/field only used by some subclasses.

**Solution:** Move to those subclasses.

```cpp
// BEFORE - get_quota only relevant to Salesman
class Employee {
public:
    virtual double get_quota() { return 0; }  // Not relevant to all
};

// AFTER
class Salesman : public Employee {
public:
    double get_quota() { return quota_; }  // Moved here
};
```

---

### Extract Subclass

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Large Class, features used only sometimes |

**Problem:** Class has features used only in certain cases.

**Solution:** Create a subclass for those cases.

```cpp
// BEFORE
class JobItem {
    double unit_price_;
    int quantity_;
    Employee* employee_;  // Only for labor items
    bool is_labor_;
public:
    double total_price() {
        return unit_price_ * quantity_;
    }
    double unit_price() {
        return is_labor_ ? employee_->rate() : unit_price_;
    }
};

// AFTER
class JobItem {
protected:
    int quantity_;
public:
    virtual double unit_price() = 0;
    double total_price() { return unit_price() * quantity_; }
};

class PartsItem : public JobItem {
    double unit_price_;
public:
    double unit_price() override { return unit_price_; }
};

class LaborItem : public JobItem {
    Employee* employee_;
public:
    double unit_price() override { return employee_->rate(); }
};
```

---

### Extract Superclass

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Duplicate Code, similar classes |

**Problem:** Two classes have similar fields and methods.

**Solution:** Create a shared superclass.

```cpp
// BEFORE
class Department {
    std::string name_;
    int staff_count_;
public:
    double total_salary() { /* ... */ }
};

class Employee {
    std::string name_;
    int id_;
    double salary_;
public:
    double annual_cost() { return salary_ * 12; }
};

// AFTER
class Party {
protected:
    std::string name_;
public:
    virtual double annual_cost() = 0;
};

class Department : public Party {
    int staff_count_;
public:
    double annual_cost() override { return total_salary(); }
};

class Employee : public Party {
    int id_;
    double salary_;
public:
    double annual_cost() override { return salary_ * 12; }
};
```

---

### Extract Interface

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Tight coupling, untestable code |

**Problem:** Multiple clients use the same part of a class, or two classes share methods.

**Solution:** Extract the shared methods into an interface.

```cpp
// BEFORE - Tightly coupled
class Employee {
public:
    double rate();
    bool has_special_skill();
    std::string name();
};

// Client depends on all of Employee

// AFTER
class IBillable {
public:
    virtual double rate() = 0;
    virtual bool has_special_skill() = 0;
    virtual ~IBillable() = default;
};

class Employee : public IBillable {
public:
    double rate() override;
    bool has_special_skill() override;
    std::string name();  // Not in interface
};

// Client depends only on IBillable - can mock for testing
```

---

### Collapse Hierarchy

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Low |
| **Fixes** | Lazy Class, unnecessary subclass |

**Problem:** Subclass is almost identical to superclass.

**Solution:** Merge them.

```cpp
// BEFORE
class Employee {
public:
    virtual double rate() { return base_rate_; }
};

class Salesman : public Employee {
    // Adds nothing meaningful
};

// AFTER
class Employee {
public:
    double rate() { return base_rate_; }
};
// Salesman class deleted
```

---

### Form Template Method

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Similar algorithms in subclasses |

**Problem:** Subclasses implement similar algorithms with same steps in same order.

**Solution:** Move algorithm structure to superclass, override variant steps.

```cpp
// BEFORE
class Site {
public:
    virtual double bill_amount() = 0;
};

class ResidentialSite : public Site {
    double bill_amount() override {
        double base = units_ * rate_;
        double tax = base * TAX_RATE;
        return base + tax;
    }
};

class LifelineSite : public Site {
    double bill_amount() override {
        double base = units_ * rate_ * 0.5;  // Different
        double tax = base * TAX_RATE * 0.2;  // Different
        return base + tax;
    }
};

// AFTER - Template Method pattern
class Site {
public:
    double bill_amount() {  // Template method
        return get_base_amount() + get_tax_amount();
    }
protected:
    virtual double get_base_amount() = 0;
    virtual double get_tax_amount() = 0;
};

class ResidentialSite : public Site {
protected:
    double get_base_amount() override { return units_ * rate_; }
    double get_tax_amount() override { return get_base_amount() * TAX_RATE; }
};

class LifelineSite : public Site {
protected:
    double get_base_amount() override { return units_ * rate_ * 0.5; }
    double get_tax_amount() override { return get_base_amount() * TAX_RATE * 0.2; }
};
```

---

### Replace Inheritance with Delegation

| Aspect | Details |
| -------- | --------- |
| **Impact** | High |
| **Risk** | Medium |
| **Fixes** | Refused Bequest, wrong hierarchy |

**Problem:** Subclass uses only part of superclass interface, or inheritance is inappropriate.

**Solution:** Create field for superclass, delegate methods, remove inheritance.

```cpp
// BEFORE - Stack shouldn't inherit from Vector
class Stack : public std::vector<int> {
public:
    void push(int element) { push_back(element); }
    int pop() {
        int result = back();
        pop_back();
        return result;
    }
    // But Stack exposes ALL vector methods!
};

// AFTER - Composition
class Stack {
    std::vector<int> data_;  // Delegation
public:
    void push(int element) { data_.push_back(element); }
    int pop() {
        int result = data_.back();
        data_.pop_back();
        return result;
    }
    bool empty() const { return data_.empty(); }
    // Only exposes stack operations
};
```

---

### Replace Delegation with Inheritance

| Aspect | Details |
| -------- | --------- |
| **Impact** | Medium |
| **Risk** | Medium |
| **Fixes** | Excessive delegation (opposite of above) |

**Problem:** Class delegates almost all methods to another class.

**Solution:** Make the delegating class inherit from the delegate.

```cpp
// BEFORE - Too much delegation
class Employee {
    Person person_;
public:
    std::string name() { return person_.name(); }
    void set_name(std::string n) { person_.set_name(n); }
    std::string address() { return person_.address(); }
    void set_address(std::string a) { person_.set_address(a); }
    std::string phone() { return person_.phone(); }
    // ... 10 more delegation methods
};

// AFTER
class Employee : public Person {
    // Inherits all Person methods directly
};
```

---

## Integration with Other Skills

| Scenario | Next Skill |
| ---------- | ------------ |
| Need to identify what to refactor | Use `code-smells` first |
| Planning a refactoring project | Use `planning-refactor` |
| Applying C++ best practices | Use `programming-cpp` |
| Using design patterns | Use `programming-cpp-design-patterns` |
| After refactoring, need tests | Use `testing-gtest-gmock` or `testing-pytest` |

## References

- [Refactoring.Guru - Refactoring Techniques](https://refactoring.guru/refactoring/techniques)
- [Refactoring: Improving the Design of Existing Code](https://martinfowler.com/books/refactoring.html) by Martin Fowler
