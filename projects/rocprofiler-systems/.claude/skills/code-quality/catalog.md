# Code Quality Catalog

Reference for writing code that does not contain these smells. Organized in two parts:

- **Part 1: Smell Detection** -- what not to write, with thresholds
- **Part 2: Techniques** -- the shape to write instead, with steps and examples

Search the tables below, then open the matching `####` heading. Do not scan the whole file. **Fixes** names are Part 2 headings, in the order to try them.

While writing code, check **every row** of the Smell index and use the listed technique so the smell is not introduced. Do not write the smelly form and report it afterward.

## Smell index

| Smell | Category | Fixes, in order |
| --- | --- | --- |
| Long Method | Bloaters | Extract Method, Replace Temp with Query, Introduce Parameter Object, Decompose Conditional, Replace Method with Method Object |
| Large Class | Bloaters | Extract Class, Extract Subclass, Extract Interface |
| Primitive Obsession | Bloaters | Replace Data Value with Object, Introduce Parameter Object, Preserve Whole Object, Replace Type Code with Class, Replace Type Code with Subclasses, Replace Type Code with State/Strategy |
| Long Parameter List | Bloaters | Replace Parameter with Method Call, Preserve Whole Object, Introduce Parameter Object |
| Data Clumps | Bloaters | Extract Class, Introduce Parameter Object, Preserve Whole Object |
| Switch Statements | Object-Orientation Abusers | Replace Conditional with Polymorphism, Replace Type Code with Subclasses, Replace Type Code with State/Strategy |
| Temporary Field | Object-Orientation Abusers | Extract Class, Replace Method with Method Object, Introduce Null Object |
| Refused Bequest | Object-Orientation Abusers | Replace Inheritance with Delegation, Extract Superclass |
| Alternative Classes with Different Interfaces | Object-Orientation Abusers | Rename Method, Move Method, Extract Superclass |
| Divergent Change | Change Preventers | Extract Class, Extract Superclass, Extract Subclass |
| Shotgun Surgery | Change Preventers | Move Method, Move Field, Inline Class |
| Parallel Inheritance Hierarchies | Change Preventers | Move Method, Move Field |
| Comments | Dispensables | Extract Variable, Extract Method, Rename Method, Introduce Assertion |
| Duplicate Code | Dispensables | Extract Method, Pull Up Method, Form Template Method, Extract Superclass, Consolidate Conditional Expression |
| Lazy Class | Dispensables | Inline Class, Collapse Hierarchy |
| Data Class | Dispensables | Encapsulate Field, Encapsulate Collection, Move Method |
| Dead Code | Dispensables | Delete, Remove Parameter, Inline Class, Collapse Hierarchy |
| Speculative Generality | Dispensables | Collapse Hierarchy, Inline Class, Inline Method, Remove Parameter |
| Feature Envy | Couplers | Move Method, Extract Method |
| Inappropriate Intimacy | Couplers | Move Method, Move Field, Extract Class, Hide Delegate, Change Bidirectional to Unidirectional |
| Message Chains | Couplers | Hide Delegate, Extract Method, Move Method |
| Middle Man | Couplers | Remove Middle Man |
| Incomplete Library Class | Couplers | Introduce Foreign Method, Introduce Local Extension |

Three names in the **Fixes** column have no Part 2 heading: **Delete** (remove the unused code), **Replace Parameter with Method Call** (get the value from an object the method can already reach), and **Change Bidirectional to Unidirectional** (drop one side of a two-way link). Apply those from the smell's Part 1 section.

## Technique index

| Technique | Category | Use for |
| --- | --- | --- |
| Extract Method | Composing Methods | Long Method, Duplicate Code, Comments |
| Inline Method | Composing Methods | Speculative Generality, excessive delegation |
| Extract Variable | Composing Methods | Complex expressions, Comments |
| Replace Temp with Query | Composing Methods | Long Method, preparation for Extract Method |
| Split Temporary Variable | Composing Methods | Variable reuse, unclear intent |
| Remove Assignments to Parameters | Composing Methods | Confusing parameter modification |
| Replace Method with Method Object | Composing Methods | Long Method with intertwined local variables |
| Substitute Algorithm | Composing Methods | Inefficient or unclear algorithm |
| Move Method | Moving Features Between Objects | Feature Envy, Shotgun Surgery |
| Move Field | Moving Features Between Objects | Feature Envy, Data Clumps |
| Extract Class | Moving Features Between Objects | Large Class, Divergent Change, Data Clumps |
| Inline Class | Moving Features Between Objects | Lazy Class, Speculative Generality |
| Hide Delegate | Moving Features Between Objects | Message Chains, tight coupling |
| Remove Middle Man | Moving Features Between Objects | Middle Man |
| Introduce Foreign Method | Moving Features Between Objects | Incomplete Library Class, 1-2 methods |
| Introduce Local Extension | Moving Features Between Objects | Incomplete Library Class, many methods |
| Self Encapsulate Field | Organizing Data | Direct field access |
| Replace Data Value with Object | Organizing Data | Primitive Obsession, Data Clumps |
| Replace Magic Number with Symbolic Constant | Organizing Data | Magic numbers |
| Encapsulate Field | Organizing Data | Public fields, Data Class |
| Encapsulate Collection | Organizing Data | Collection exposure |
| Replace Type Code with Class | Organizing Data | Primitive Obsession, type codes without behavior |
| Replace Type Code with Subclasses | Organizing Data | Switch Statements, type codes with behavior |
| Replace Type Code with State/Strategy | Organizing Data | Switch Statements when the type changes at runtime |
| Decompose Conditional | Simplifying Conditional Expressions | Long Method, complex conditionals |
| Consolidate Conditional Expression | Simplifying Conditional Expressions | Duplicate conditionals |
| Consolidate Duplicate Conditional Fragments | Simplifying Conditional Expressions | Duplicate Code in branches |
| Replace Nested Conditional with Guard Clauses | Simplifying Conditional Expressions | Deep nesting |
| Replace Conditional with Polymorphism | Simplifying Conditional Expressions | Switch Statements, type-based conditionals |
| Introduce Null Object | Simplifying Conditional Expressions | Repeated null checks |
| Introduce Assertion | Simplifying Conditional Expressions | Implicit assumptions |
| Rename Method | Simplifying Method Calls | Unclear intent, Comments |
| Add Parameter / Remove Parameter | Simplifying Method Calls | Missing or unused data |
| Separate Query from Modifier | Simplifying Method Calls | Side effects, hard to test |
| Parameterize Method | Simplifying Method Calls | Similar methods, Duplicate Code |
| Introduce Parameter Object | Simplifying Method Calls | Long Parameter List, Data Clumps |
| Preserve Whole Object | Simplifying Method Calls | Long Parameter List, Feature Envy |
| Replace Constructor with Factory Method | Simplifying Method Calls | Complex construction, type-based creation |
| Replace Error Code with Exception | Simplifying Method Calls | Error-handling clutter |
| Replace Exception with Test | Simplifying Method Calls | Exception misuse |
| Pull Up Field / Pull Up Method | Dealing with Generalization | Duplicate Code in subclasses |
| Pull Up Constructor Body | Dealing with Generalization | Duplicate constructor code |
| Push Down Method / Push Down Field | Dealing with Generalization | Superclass has irrelevant members |
| Extract Subclass | Dealing with Generalization | Large Class, features used only sometimes |
| Extract Superclass | Dealing with Generalization | Duplicate Code, similar classes |
| Extract Interface | Dealing with Generalization | Tight coupling, untestable code |
| Collapse Hierarchy | Dealing with Generalization | Lazy Class, unnecessary subclass |
| Form Template Method | Dealing with Generalization | Similar algorithms in subclasses |
| Replace Inheritance with Delegation | Dealing with Generalization | Refused Bequest, wrong hierarchy |
| Replace Delegation with Inheritance | Dealing with Generalization | Excessive delegation |

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

## Part 2: Techniques

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
    return m_late_deliveries > 5;
}

// AFTER
int get_rating() {
    return m_late_deliveries > 5 ? 2 : 1;
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
    double base_price = m_quantity * m_item_price;
    if (base_price > 1000) {
        return base_price * 0.95;
    }
    return base_price * 0.98;
}

// AFTER
double base_price() const {
    return m_quantity * m_item_price;
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
double temp = 2 * (m_height + m_width);
std::cout << "Perimeter: " << temp << "\n";
temp = m_height * m_width;
std::cout << "Area: " << temp << "\n";

// AFTER
double perimeter = 2 * (m_height + m_width);
std::cout << "Perimeter: " << perimeter << "\n";
double area = m_height * m_width;
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
    Order& m_order;
    double m_primary_base;
    double m_secondary;
    double m_tertiary;

public:
    PriceCalculator(Order& order) : m_order(order) {}

    double compute() {
        m_primary_base = // complex calculation
        m_secondary = // uses m_primary_base
        m_tertiary = // uses m_secondary
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
    AccountType m_type;
public:
    double overdraft_charge() {
        if (m_type.is_premium()) {
            double result = 10;
            if (m_days_overdrawn > 7) {
                result += (m_days_overdrawn - 7) * 0.85;
            }
            return result;
        }
        return m_days_overdrawn * 1.75;
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
        return m_type.overdraft_charge(m_days_overdrawn);
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
    AccountType m_type;
    double m_interest_rate;
};

// AFTER - interest_rate moved to AccountType
class AccountType {
    double m_interest_rate;
public:
    double interest_rate() const { return m_interest_rate; }
    void set_interest_rate(double rate) { m_interest_rate = rate; }
};

class Account {
    AccountType m_type;
public:
    double interest_rate() const { return m_type.interest_rate(); }
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
    std::string m_name;
    std::string m_office_area_code;
    std::string m_office_number;
public:
    std::string telephone_number() {
        return "(" + m_office_area_code + ") " + m_office_number;
    }
};

// AFTER
class TelephoneNumber {
    std::string m_area_code;
    std::string m_number;
public:
    std::string format() const {
        return "(" + m_area_code + ") " + m_number;
    }
};

class Person {
    std::string m_name;
    TelephoneNumber m_office_telephone;
public:
    std::string telephone_number() {
        return m_office_telephone.format();
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
    std::string m_number;
public:
    std::string number() const { return m_number; }
};

class Person {
    TelephoneNumber m_phone;
public:
    std::string phone_number() { return m_phone.number(); }
};

// AFTER
class Person {
    std::string m_phone_number;
public:
    std::string phone_number() const { return m_phone_number; }
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
    Department* m_department;
public:
    Department* department() { return m_department; }
};
// Client: manager = john.department()->manager();

// AFTER - Department hidden
class Person {
    Department* m_department;
public:
    Person* manager() { return m_department->manager(); }
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
    Department* m_department;
public:
    Person* manager() { return m_department->manager(); }
    std::string budget() { return m_department->budget(); }
    std::string team_size() { return m_department->team_size(); }
    // ... 10 more delegation methods
};

// AFTER - Expose delegate
class Person {
    Department* m_department;
public:
    Department* department() { return m_department; }
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
    Date m_date;
public:
    explicit DateWrapper(const Date& d) : m_date(d) {}
    int year() const { return m_date.year(); }
    int month() const { return m_date.month(); }
    int day() const { return m_date.day(); }
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
    int m_low, m_high;
public:
    bool includes(int arg) {
        return arg >= m_low && arg <= m_high;  // Direct access
    }
};

// AFTER
class IntRange {
    int m_low, m_high;
public:
    int low() const { return m_low; }
    int high() const { return m_high; }
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
    std::string m_customer_name;
    std::string m_customer_email;
    std::string m_customer_phone;
};

// AFTER
class Customer {
    std::string m_name;
    std::string m_email;
    std::string m_phone;
public:
    bool is_valid_email() const;
    std::string formatted_phone() const;
};

class Order {
    Customer m_customer;
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
constexpr double k_gravitational_acceleration = 9.81;  // m/s^2

double potential_energy(double mass, double height) {
    return mass * k_gravitational_acceleration * height;
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
    std::string m_name;
public:
    const std::string& name() const { return m_name; }
    void set_name(const std::string& name) { m_name = name; }
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
    std::vector<Course> m_courses;
public:
    std::vector<Course>& courses() { return m_courses; }  // Dangerous!
};

// AFTER
class Person {
    std::vector<Course> m_courses;
public:
    const std::vector<Course>& courses() const { return m_courses; }
    void add_course(const Course& c) { m_courses.push_back(c); }
    void remove_course(const Course& c) {
        m_courses.erase(
            std::remove(m_courses.begin(), m_courses.end(), c),
            m_courses.end());
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
    int m_blood_type;  // 0=O, 1=A, 2=B, 3=AB
    static constexpr int k_o = 0;
    static constexpr int k_a = 1;
};

// AFTER
class BloodType {
    int m_code;
    BloodType(int code) : m_code(code) {}
public:
    static const BloodType k_o;
    static const BloodType k_a;
    static const BloodType k_b;
    static const BloodType k_ab;
    bool operator==(const BloodType& other) const {
        return m_code == other.m_code;
    }
};

class Person {
    BloodType m_blood_type;
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
    int m_type;
public:
    int bonus() {
        switch (m_type) {
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
    int m_type;
public:
    int pay_amount() {
        switch (m_type) {
            case ENGINEER: return m_monthly_salary;
            case SALESMAN: return m_monthly_salary + m_commission;
            case MANAGER: return m_monthly_salary + m_bonus;
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
    std::unique_ptr<EmployeeType> m_type;
public:
    void set_type(std::unique_ptr<EmployeeType> t) {
        m_type = std::move(t);  // Can change at runtime
    }
    int pay_amount() { return m_type->pay_amount(*this); }
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
        return quantity * m_winter_rate + m_winter_service_charge;
    } else {
        return quantity * m_summer_rate;
    }
}

// AFTER
bool is_summer(const Date& date) {
    return date >= SUMMER_START && date <= SUMMER_END;
}
double summer_charge(int quantity) { return quantity * m_summer_rate; }
double winter_charge(int quantity) {
    return quantity * m_winter_rate + m_winter_service_charge;
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
    if (m_seniority < 2) return 0;
    if (m_months_disabled > 12) return 0;
    if (m_is_part_time) return 0;
    return calculate_disability();
}

// AFTER
bool is_not_eligible_for_disability() {
    return m_seniority < 2 || m_months_disabled > 12 || m_is_part_time;
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
    if (m_is_dead) {
        result = dead_amount();
    } else {
        if (m_is_separated) {
            result = separated_amount();
        } else {
            if (m_is_retired) {
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
    if (m_is_dead) return dead_amount();
    if (m_is_separated) return separated_amount();
    if (m_is_retired) return retired_amount();
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
    std::string m_type;
public:
    double get_speed() {
        switch (m_type) {
            case "European": return get_base_speed();
            case "African":
                return get_base_speed() - m_load_factor * m_num_coconuts;
            case "NorwegianBlue":
                return m_is_nailed ? 0 : get_base_speed(m_voltage);
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
        return get_base_speed() - m_load_factor * m_num_coconuts;
    }
};
class NorwegianBlue : public Bird {
public:
    double get_speed() override {
        return m_is_nailed ? 0 : get_base_speed(m_voltage);
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
    return (m_expense_limit != k_null_expense)
        ? m_expense_limit
        : m_primary_project->member_expense_limit();
}

// AFTER
double get_expense_limit() {
    assert(m_expense_limit != k_null_expense || m_primary_project != nullptr);
    return (m_expense_limit != k_null_expense)
        ? m_expense_limit
        : m_primary_project->member_expense_limit();
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
std::string get_tlp() { return m_telephone; }
// AFTER
std::string get_telephone_number() { return m_telephone; }
```

---

#### Add Parameter / Remove Parameter

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Medium | Missing/unused data |

```cpp
// Remove Parameter
// BEFORE
void set_value(int value, int unused) { m_value = value; }
// AFTER
void set_value(int value) { m_value = value; }
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
void five_percent_raise() { m_salary *= 1.05; }
void ten_percent_raise() { m_salary *= 1.10; }

// AFTER
void raise(double factor) { m_salary *= (1 + factor); }
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
    Date m_start, m_end;
public:
    DateRange(Date start, Date end) : m_start(start), m_end(end) {}
    bool includes(Date d) const { return d >= m_start && d <= m_end; }
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
    if (amount > m_balance) return -1;  // Error code
    m_balance -= amount;
    return 0;
}

// AFTER
void withdraw(int amount) {
    if (amount > m_balance) {
        throw std::runtime_error("Insufficient funds");
    }
    m_balance -= amount;
}
```

---

#### Replace Exception with Test

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Exception misuse |

```cpp
// BEFORE
double m_valuefor_period(int period) {
    try {
        return m_values.at(period);
    } catch (std::out_of_range&) {
        return 0;
    }
}

// AFTER
double m_valuefor_period(int period) {
    if (period < 0 || period >= m_values.size()) return 0;
    return m_values[period];
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
class Salesman : public Employee { std::string m_name; };  // Duplicated
class Engineer : public Employee { std::string m_name; };  // Duplicated

// AFTER
class Employee {
protected:
    std::string m_name;  // Moved to parent
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
        m_name = name; m_id = id; m_grade = grade;
    }
};

// AFTER
class Employee {
protected:
    Employee(std::string name, std::string id)
        : m_name(name), m_id(id) {}
};
class Manager : public Employee {
public:
    Manager(std::string name, std::string id, int grade)
        : Employee(name, id), m_grade(grade) {}
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
    double get_quota() { return m_quota; }  // Moved here
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
    double m_unit_price;
    Employee* m_employee;  // Only for labor items
    bool m_is_labor;
public:
    double unit_price() {
        return m_is_labor ? m_employee->rate() : m_unit_price;
    }
};

// AFTER
class JobItem {
public:
    virtual double unit_price() = 0;
};
class PartsItem : public JobItem {
    double m_unit_price;
public:
    double unit_price() override { return m_unit_price; }
};
class LaborItem : public JobItem {
    Employee* m_employee;
public:
    double unit_price() override { return m_employee->rate(); }
};
```

---

#### Extract Superclass

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Duplicate Code, similar classes |

```cpp
// BEFORE -- Department and Employee both have m_name and annual cost
class Department { std::string m_name; /* ... */ };
class Employee { std::string m_name; /* ... */ };

// AFTER
class Party {
protected:
    std::string m_name;
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
class Employee { virtual double rate() { return m_base_rate; } };
class Salesman : public Employee { /* adds nothing */ };

// AFTER -- Salesman class deleted
class Employee { double rate() { return m_base_rate; } };
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
    double get_base_amount() override { return m_units * m_rate; }
    double get_tax_amount() override { return get_base_amount() * TAX_RATE; }
};

class LifelineSite : public Site {
protected:
    double get_base_amount() override { return m_units * m_rate * 0.5; }
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
    std::vector<int> m_data;
public:
    void push(int element) { m_data.push_back(element); }
    int pop() {
        int result = m_data.back();
        m_data.pop_back();
        return result;
    }
    bool empty() const { return m_data.empty(); }
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
    Person m_person;
public:
    std::string name() { return m_person.name(); }
    void set_name(std::string n) { m_person.set_name(n); }
    std::string address() { return m_person.address(); }
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
