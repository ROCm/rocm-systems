# Code Quality Catalog

Full reference of 22 code smells and 60+ refactoring techniques. Organized in two parts:

- **Part 1: Smell Detection** -- what to look for, thresholds, severity
- **Part 2: Refactoring Techniques** -- how to fix each smell, with step-by-step instructions and code examples

---

## Part 1: Smell Detection

### Category 1: Bloaters

Code that has grown to excessive proportions.

#### Long Method

**Threshold:** >10 lines warrants questions, >50 lines is definite smell

**Signs:**

- Method contains too many lines of code
- Method does multiple things
- Requires extensive comments to explain

**Detection criteria:**

```text
- Lines > 50: Should Fix (50)
- Lines > 100: Must Fix (80)
- Method does more than one thing (multiple responsibilities)
- Requires extensive comments to explain
```

**Fixes:** Extract Method, Replace Temp with Query, Introduce Parameter Object, Decompose Conditional, Replace Method with Method Object

---

#### Large Class

**Threshold:** >500 lines warrants review

**Signs:**

- Class contains excessive fields/methods/lines
- Class has multiple unrelated responsibilities
- You struggle to summarize what the class does in one sentence

**Detection criteria:**

```text
- Lines > 500: Should Fix (50)
- Lines > 1000: Must Fix (80)
- >10 public methods: Review for SRP violation
- >15 fields: Likely doing too much
```

**Fixes:** Extract Class, Extract Subclass, Extract Interface

---

#### Primitive Obsession

**Signs:**

- Using primitives instead of small objects (money, phone numbers, ranges)
- Constants encoding information (`USER_ADMIN_ROLE = 1`)
- Using `int` for IDs instead of typed wrapper

**Detection criteria:**

```text
- Multiple functions operating on same primitive group: Should Fix (50)
- Constants simulating types: Should Fix (50)
- No domain objects for business concepts: Should Fix (50)
```

**Fixes:** Replace Data Value with Object, Introduce Parameter Object, Preserve Whole Object, Replace Type Code with Class/Subclasses/State

---

#### Long Parameter List

**Threshold:** >3-4 parameters

**Detection criteria:**

```text
- 4-5 parameters: Should Fix (50)
- 6+ parameters: Must Fix (80)
- Boolean "flag" parameters: Should Fix (50)
- Parameters from same object: Should Fix (50)
```

**Fixes:** Replace Parameter with Method Call, Preserve Whole Object, Introduce Parameter Object

**When to ignore:** Creating dependencies between classes may be worse than long parameter list.

---

#### Data Clumps

**Signs:**

- Same group of variables appears in multiple places
- Parameters that always travel together
- Removing one variable makes others meaningless

**Detection criteria:**

```text
- Same 3+ parameters in multiple method signatures: Should Fix (50)
- Same fields grouped in multiple classes: Should Fix (50)
```

**Fixes:** Extract Class, Introduce Parameter Object, Preserve Whole Object

---

### Category 2: Object-Orientation Abusers

Incomplete or incorrect application of OOP principles.

#### Switch Statements

**Signs:**

- Complex `switch` or `if-else` chain based on type
- Same switch logic scattered across multiple methods
- Adding a new case requires changes in many places

**Detection criteria:**

```text
- Switch on type/enum with behavior: Should Fix (50)
- Same switch in multiple places: Must Fix (80)
- Switch with >5 cases with different behavior: Should Fix (50)
```

**Fixes:** Replace Conditional with Polymorphism, Replace Type Code with Subclasses, Replace Type Code with State/Strategy

**When to ignore:** Factory patterns legitimately use switch to create objects.

---

#### Temporary Field

**Signs:**

- Fields only used in certain circumstances
- Fields remain empty/null most of the time

**Detection criteria:**

```text
- Field used in <25% of methods: Should Fix (50)
- Field only set in one method and used in another: Should Fix (50)
```

**Fixes:** Extract Class, Replace Method with Method Object, Introduce Null Object

---

#### Refused Bequest

**Signs:**

- Subclass uses only some parent methods/properties
- Inherited methods throw exceptions or do nothing
- Subclass doesn't follow Liskov Substitution Principle

**Detection criteria:**

```text
- Override methods with empty body or exception: Should Fix (50)
- Subclass ignores >50% of parent interface: Should Fix (50)
- Inheritance for code reuse, not "is-a": Should Fix (50)
```

**Fixes:** Replace Inheritance with Delegation, Extract Superclass

---

#### Alternative Classes with Different Interfaces

**Signs:**

- Two classes do the same thing with different method names
- Duplicate functionality discovered during review

**Detection criteria:**

```text
- Classes with same purpose, different interface: Should Fix (50)
- Duplicate logic in differently-named methods: Should Fix (50)
```

**Fixes:** Rename Method, Move Method, Extract Superclass

**When to ignore:** Classes in different libraries with independent versioning.

---

### Category 3: Change Preventers

Issues requiring changes in multiple places for a single change.

#### Divergent Change

**Signs:**

- Single class needs changes for multiple unrelated reasons
- Class has multiple "areas" of responsibility

**Detection criteria:**

```text
- Class changes for >2 unrelated reasons: Should Fix (50)
- Methods naturally group into separate concerns: Should Fix (50)
```

**Fixes:** Extract Class, Extract Superclass/Subclass

---

#### Shotgun Surgery

**Signs:**

- Single change requires edits to many different classes
- Adding a feature means touching 5+ files

**Detection criteria:**

```text
- Feature addition touches 5+ classes: Must Fix (80)
- Single responsibility scattered across classes: Must Fix (80)
```

**Fixes:** Move Method/Field, Inline Class

---

#### Parallel Inheritance Hierarchies

**Signs:**

- Creating subclass in one hierarchy requires subclass in another
- Mirrored class structures

**Detection criteria:**

```text
- Two hierarchies with matching structures: Should Fix (50)
- Adding to one requires adding to another: Should Fix (50)
```

**Fixes:** Move Method/Field to consolidate; eliminate redundant hierarchy

**When to ignore:** If deduplication makes code uglier, keep the parallel structure.

---

### Category 4: Dispensables

Pointless elements whose absence would make code cleaner.

#### Comments

**Detection criteria:**

```text
- Comment explains what code does (not why): Nitpick (20)
- Commented-out code: Should Fix (50)
- Comment contradicts code: Must Fix (80)
- TODOs without ticket references: Nitpick (20)
```

**Fixes:** Extract Variable, Extract Method, Rename Method, Introduce Assertion

**When acceptable:** Comments explaining WHY (design decisions, complex algorithms, non-obvious constraints).

---

#### Duplicate Code

**Detection criteria:**

```text
- Identical code blocks: Should Fix (50)
- Similar code blocks (>80% same): Should Fix (50)
- Copy-paste with minor modifications: Should Fix (50)
```

**Fixes:** Extract Method, Pull Up Method, Form Template Method, Extract Superclass, Consolidate Conditional Expression

---

#### Lazy Class

**Detection criteria:**

```text
- Class with <3 methods: Nitpick (20)
- Class that just delegates to another: Nitpick (20)
- Subclass with minimal additions: Nitpick (20)
```

**Fixes:** Inline Class, Collapse Hierarchy

**When to ignore:** Class represents future expansion point.

---

#### Data Class

**Signs:**

- Class contains only fields and getters/setters
- No behavior, just data container

**Detection criteria:**

```text
- Class with only fields + accessors: Should Fix (50)
- No methods operating on own data: Should Fix (50)
```

**Fixes:** Encapsulate Field, Encapsulate Collection, Move Method (bring behavior into class)

---

#### Dead Code

**Detection criteria:**

```text
- Unused variable: Nitpick (20)
- Unused method/function: Should Fix (50)
- Unreachable code: Should Fix (50)
- Commented-out code: Should Fix (50)
```

**Fixes:** Delete, Remove Parameter, Inline Class/Collapse Hierarchy

---

#### Speculative Generality

**Signs:**

- Unused class, method, field, or parameter
- Hooks for future features that never came
- Overly abstract code for simple requirements

**Detection criteria:**

```text
- Unused abstraction: Nitpick (20)
- Abstract class with single implementation: Nitpick (20)
- Methods that only delegate: Nitpick (20)
```

**Fixes:** Collapse Hierarchy, Inline Class, Inline Method, Remove Parameter

**When to ignore:** Framework code where users may need the hooks.

---

### Category 5: Couplers

Excessive coupling between classes or excessive delegation.

#### Feature Envy

**Detection criteria:**

```text
- Method uses >3 getters from another object: Should Fix (50)
- Method primarily operates on another class's data: Should Fix (50)
```

**Fixes:** Move Method, Extract Method (move only envious part)

**When to ignore:** Strategy/Visitor patterns intentionally separate behavior from data.

---

#### Inappropriate Intimacy

**Detection criteria:**

```text
- Classes accessing each other's private/protected: Must Fix (80)
- Bidirectional association: Should Fix (50)
- Deep knowledge of another class's internals: Should Fix (50)
```

**Fixes:** Move Method/Field, Extract Class, Hide Delegate, Change Bidirectional to Unidirectional

---

#### Message Chains

**Signs:**

- Series of calls like `a.b().c().d()`
- Client depends on navigation structure

**Detection criteria:**

```text
- Chain of 3+ method calls: Should Fix (50)
- "Train wreck" code: Should Fix (50)
```

**Fixes:** Hide Delegate, Extract Method + Move Method

**Caution:** Overly aggressive hiding creates Middle Man smell.

---

#### Middle Man

**Detection criteria:**

```text
- Class where >50% methods just delegate: Should Fix (50)
- Class with only pass-through methods: Should Fix (50)
```

**Fixes:** Remove Middle Man

**When to ignore:** Proxy, Decorator, or testing isolation patterns.

---

#### Incomplete Library Class

**Detection criteria:**

```text
- Multiple workarounds for same library limitation: Should Fix (50)
- Helper functions for library functionality: Nitpick (20)
```

**Fixes:** Introduce Foreign Method (few methods), Introduce Local Extension (many methods)

---
---

## Part 2: Refactoring Techniques

### Category A: Composing Methods

#### Extract Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Low | Long Method, Duplicate Code, Comments |

**Problem:** Code fragment that can be logically grouped together.
**Solution:** Create a new method named after WHAT it does and move the fragment there.

**Steps:**

1. Create new method named after WHAT it does (not HOW)
2. Copy the code fragment to the new method
3. Look for local variables used only within the fragment -- make them local to new method
4. Pass external variables as parameters
5. If extracted code modifies a variable needed later, return it
6. Replace original fragment with method call

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

---

#### Inline Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Low | Low | Speculative Generality, excessive delegation |

**Problem:** Method body is more obvious than the method itself.
**Solution:** Replace method calls with the method content and delete the method.

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

#### Extract Variable

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Complex expressions, Comments |

**Problem:** Hard-to-understand expression.
**Solution:** Place the expression result in a self-explanatory variable.

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

#### Replace Temp with Query

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Long Method (preparation for Extract Method) |

**Problem:** Temporary variable storing an expression result for later use.
**Solution:** Move the expression to a new method and call the method instead.

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

#### Split Temporary Variable

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Variable reuse, unclear intent |

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

#### Remove Assignments to Parameters

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Confusing parameter modification |

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

#### Replace Method with Method Object

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Long Method with intertwined local variables |

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

#### Substitute Algorithm

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | High | Inefficient or unclear algorithm |

**Problem:** Want to replace an algorithm with a better one.
**Solution:** Replace the method body with the new algorithm.

**Steps:**

1. Simplify the existing algorithm first
2. Create tests covering all edge cases
3. Write the new algorithm
4. Run tests to verify behavior is identical

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

### Category B: Moving Features Between Objects

#### Move Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Feature Envy, Shotgun Surgery |

**Problem:** Method is used more by another class than its own class.
**Solution:** Move the method to the class that uses it most.

**Steps:**

1. Examine all features used by the method in its current class
2. Check if method is declared in parent/child classes
3. Declare the method in the target class
4. Copy the method body, adapting references to the new context
5. Convert original method to delegate or remove it
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

#### Move Field

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Feature Envy, Data Clumps |

**Problem:** Field is used more by another class than its own class.
**Solution:** Create a field in the target class and redirect all users.

**Steps:**

1. If field is public, encapsulate it first
2. Create field and accessors in target class
3. Update all references to use the new location
4. Remove the field from the original class

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

#### Extract Class

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Large Class, Divergent Change, Data Clumps |

**Problem:** One class does the work of two.
**Solution:** Create a new class and move relevant fields and methods.

**Steps:**

1. Decide what to split out
2. Create a new class for the extracted responsibility
3. Create a link from the old class to the new one
4. Use Move Field for each field to move
5. Use Move Method for each method to move
6. Review and reduce interfaces of both classes

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

#### Inline Class

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Lazy Class, Speculative Generality |

**Problem:** Class does almost nothing and has no planned responsibilities.
**Solution:** Move all features to another class and delete the empty class.

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

#### Hide Delegate

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Message Chains, tight coupling |

**Problem:** Client calls object B from object A, then calls methods on B.
**Solution:** Create a method in A that delegates to B, hiding B from the client.

```cpp
// BEFORE - Client knows about Department
class Person {
    Department* department_;
public:
    Department* department() { return department_; }
};
// Client: manager = john.department()->manager();

// AFTER - Department hidden
class Person {
    Department* department_;
public:
    Person* manager() { return department_->manager(); }
};
// Client: manager = john.manager();
```

---

#### Remove Middle Man

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Middle Man |

**Problem:** Class has too many methods that simply delegate.
**Solution:** Delete delegation methods; have clients call the delegate directly.

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

// AFTER - Expose delegate
class Person {
    Department* department_;
public:
    Department* department() { return department_; }
};
// Client: auto budget = john.department()->budget();
```

---

#### Introduce Foreign Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Low | Low | Incomplete Library Class (1-2 methods) |

**Problem:** Utility class lacks a method you need, and you cannot modify it.
**Solution:** Create the method in your client class, taking the utility object as parameter.

```cpp
// Can't modify Date class
class Report {
    // Foreign method - ideally should be in Date class
    static Date next_day(const Date& date) {
        return Date(date.year(), date.month(), date.day() + 1);
    }
};
```

---

#### Introduce Local Extension

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Medium | Incomplete Library Class (many methods) |

**Problem:** Utility class lacks several methods you need.
**Solution:** Create a new class (subclass or wrapper) containing the methods.

```cpp
// Subclass approach
class ExtendedDate : public Date {
public:
    using Date::Date;
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
    int year() const { return date_.year(); }
    int month() const { return date_.month(); }
    int day() const { return date_.day(); }
    Date next_day() const {
        return Date(year(), month(), day() + 1);
    }
};
```

---

### Category C: Organizing Data

#### Self Encapsulate Field

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Low | Low | Direct field access issues |

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

#### Replace Data Value with Object

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Primitive Obsession, Data Clumps |

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
    bool is_valid_email() const;
    std::string formatted_phone() const;
};

class Order {
    Customer customer_;
};
```

---

#### Replace Magic Number with Symbolic Constant

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Magic Numbers |

```cpp
// BEFORE
double potential_energy(double mass, double height) {
    return mass * 9.81 * height;
}

// AFTER
constexpr double GRAVITATIONAL_ACCELERATION = 9.81;  // m/s^2

double potential_energy(double mass, double height) {
    return mass * GRAVITATIONAL_ACCELERATION * height;
}
```

---

#### Encapsulate Field

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Public fields, Data Class |

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

#### Encapsulate Collection

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Collection exposure, broken encapsulation |

**Problem:** Method returns a mutable collection reference.
**Solution:** Return read-only view and provide add/remove methods.

```cpp
// BEFORE
class Person {
    std::vector<Course> courses_;
public:
    std::vector<Course>& courses() { return courses_; }  // Dangerous!
};

// AFTER
class Person {
    std::vector<Course> courses_;
public:
    const std::vector<Course>& courses() const { return courses_; }
    void add_course(const Course& c) { courses_.push_back(c); }
    void remove_course(const Course& c) {
        courses_.erase(
            std::remove(courses_.begin(), courses_.end(), c),
            courses_.end());
    }
};
```

---

#### Replace Type Code with Class

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Primitive Obsession (type codes without behavior) |

```cpp
// BEFORE
class Person {
    int blood_type_;  // 0=O, 1=A, 2=B, 3=AB
    static constexpr int O = 0;
    static constexpr int A = 1;
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

class Person {
    BloodType blood_type_;
};
```

---

#### Replace Type Code with Subclasses

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | High | Switch Statements, Primitive Obsession (type codes with behavior) |

```cpp
// BEFORE
class Employee {
    int type_;
public:
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
```

---

#### Replace Type Code with State/Strategy

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | High | Switch Statements when subclassing is impossible (type changes at runtime) |

```cpp
// BEFORE
class Employee {
    int type_;
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
        type_ = std::move(t);  // Can change at runtime
    }
    int pay_amount() { return type_->pay_amount(*this); }
};
```

---

### Category D: Simplifying Conditional Expressions

#### Decompose Conditional

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Low | Long Method, Complex conditionals |

**Problem:** Complex conditional (if-then-else).
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
double summer_charge(int quantity) { return quantity * summer_rate_; }
double winter_charge(int quantity) {
    return quantity * winter_rate_ + winter_service_charge_;
}
double calculate_charge(const Date& date, int quantity) {
    return is_summer(date) ? summer_charge(quantity) : winter_charge(quantity);
}
```

---

#### Consolidate Conditional Expression

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Duplicate conditionals |

**Problem:** Multiple conditionals leading to the same result.
**Solution:** Combine into a single expression and extract to a method.

```cpp
// BEFORE
double disability_amount() {
    if (seniority_ < 2) return 0;
    if (months_disabled_ > 12) return 0;
    if (is_part_time_) return 0;
    return calculate_disability();
}

// AFTER
bool is_not_eligible_for_disability() {
    return seniority_ < 2 || months_disabled_ > 12 || is_part_time_;
}
double disability_amount() {
    if (is_not_eligible_for_disability()) return 0;
    return calculate_disability();
}
```

---

#### Consolidate Duplicate Conditional Fragments

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Duplicate Code in branches |

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
total = is_special_deal() ? price * 0.95 : price * 0.98;
send_notification();  // Moved outside
```

---

#### Replace Nested Conditional with Guard Clauses

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Low | Deep nesting, unclear control flow |

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

#### Replace Conditional with Polymorphism

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | High | Switch Statements, Type-based conditionals |

**Steps:**

1. If conditional is part of larger method, use Extract Method first
2. Create subclasses for each conditional branch
3. Create abstract method in superclass
4. Override method in each subclass with branch logic
5. Delete the conditional; make method abstract

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
    double get_speed() override { return get_base_speed(); }
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

#### Introduce Null Object

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Repeated null checks |

**Problem:** Many null checks throughout the code.
**Solution:** Return a null object that exhibits default behavior.

```cpp
// BEFORE -- scattered null checks
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
// Usage -- no null checks:
plan = customer->plan();
```

---

#### Introduce Assertion

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Low | Low | Implicit assumptions |

**Problem:** Code assumes certain conditions are true.
**Solution:** Make assumptions explicit with assertions.

```cpp
// BEFORE
double get_expense_limit() {
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

### Category E: Simplifying Method Calls

#### Rename Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Low | Unclear intent, Comments |

```cpp
// BEFORE
std::string get_tlp() { return telephone_; }
// AFTER
std::string get_telephone_number() { return telephone_; }
```

---

#### Add Parameter / Remove Parameter

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Medium | Missing/unused data |

```cpp
// Remove Parameter
// BEFORE
void set_value(int value, int unused) { value_ = value; }
// AFTER
void set_value(int value) { value_ = value; }
```

---

#### Separate Query from Modifier

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Side effects, hard to test |

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
int get_total() const { return calculate_total(); }  // Query
void clear() { clear_items(); }                       // Modifier
```

---

#### Parameterize Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Similar methods, Duplicate Code |

```cpp
// BEFORE
void five_percent_raise() { salary_ *= 1.05; }
void ten_percent_raise() { salary_ *= 1.10; }

// AFTER
void raise(double factor) { salary_ *= (1 + factor); }
```

---

#### Introduce Parameter Object

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Long Parameter List, Data Clumps |

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
    bool includes(Date d) const { return d >= start_ && d <= end_; }
};

void amount_invoiced(DateRange range);
void amount_received(DateRange range);
void amount_overdue(DateRange range);
```

---

#### Preserve Whole Object

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Long Parameter List, Feature Envy |

```cpp
// BEFORE
int low = days_temp_range.low();
int high = days_temp_range.high();
bool within_plan = plan.within_range(low, high);

// AFTER
bool within_plan = plan.within_range(days_temp_range);
```

---

#### Replace Constructor with Factory Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Complex construction, Type-based creation |

```cpp
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

#### Replace Error Code with Exception

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Error handling clutter |

```cpp
// BEFORE
int withdraw(int amount) {
    if (amount > balance_) return -1;  // Error code
    balance_ -= amount;
    return 0;
}

// AFTER
void withdraw(int amount) {
    if (amount > balance_) {
        throw std::runtime_error("Insufficient funds");
    }
    balance_ -= amount;
}
```

---

#### Replace Exception with Test

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Exception misuse |

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
    if (period < 0 || period >= values_.size()) return 0;
    return values_[period];
}
```

---

### Category F: Dealing with Generalization

#### Pull Up Field / Pull Up Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Duplicate Code in subclasses |

```cpp
// BEFORE
class Salesman : public Employee { std::string name_; };  // Duplicated
class Engineer : public Employee { std::string name_; };  // Duplicated

// AFTER
class Employee {
protected:
    std::string name_;  // Moved to parent
};
```

---

#### Pull Up Constructor Body

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Duplicate constructor code |

```cpp
// BEFORE -- duplicate init in each subclass
class Manager : public Employee {
public:
    Manager(std::string name, std::string id, int grade) {
        name_ = name; id_ = id; grade_ = grade;
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

#### Push Down Method / Push Down Field

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Superclass has irrelevant members |

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

#### Extract Subclass

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Large Class, features used only sometimes |

```cpp
// BEFORE
class JobItem {
    double unit_price_;
    Employee* employee_;  // Only for labor items
    bool is_labor_;
public:
    double unit_price() {
        return is_labor_ ? employee_->rate() : unit_price_;
    }
};

// AFTER
class JobItem {
public:
    virtual double unit_price() = 0;
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

#### Extract Superclass

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Duplicate Code, similar classes |

```cpp
// BEFORE -- Department and Employee both have name_ and annual cost
class Department { std::string name_; /* ... */ };
class Employee { std::string name_; /* ... */ };

// AFTER
class Party {
protected:
    std::string name_;
public:
    virtual double annual_cost() = 0;
};
class Department : public Party { /* ... */ };
class Employee : public Party { /* ... */ };
```

---

#### Extract Interface

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Tight coupling, untestable code |

```cpp
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
};
// Client depends only on IBillable -- can mock for testing
```

---

#### Collapse Hierarchy

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Lazy Class, unnecessary subclass |

```cpp
// BEFORE
class Employee { virtual double rate() { return base_rate_; } };
class Salesman : public Employee { /* adds nothing */ };

// AFTER -- Salesman class deleted
class Employee { double rate() { return base_rate_; } };
```

---

#### Form Template Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Similar algorithms in subclasses |

```cpp
// AFTER -- Template Method pattern
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

#### Replace Inheritance with Delegation

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Refused Bequest, wrong hierarchy |

```cpp
// BEFORE - Stack shouldn't inherit from Vector
class Stack : public std::vector<int> {
    // Exposes ALL vector methods!
};

// AFTER - Composition
class Stack {
    std::vector<int> data_;
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

#### Replace Delegation with Inheritance

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Medium | Excessive delegation (opposite of above) |

```cpp
// BEFORE - Too much delegation
class Employee {
    Person person_;
public:
    std::string name() { return person_.name(); }
    void set_name(std::string n) { person_.set_name(n); }
    std::string address() { return person_.address(); }
    // ... 10 more delegation methods
};

// AFTER
class Employee : public Person {
    // Inherits all Person methods directly
};
```

---

### References

- [Refactoring.Guru - Code Smells](https://refactoring.guru/refactoring/smells)
- [Refactoring.Guru - Refactoring Techniques](https://refactoring.guru/refactoring/techniques)
- Martin Fowler, *Refactoring: Improving the Design of Existing Code*
