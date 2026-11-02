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
void
print_invoice(const invoice& inv)
{
    // Print header
    LOG_INFO("================");
    LOG_INFO("INVOICE #{}", inv.number);
    LOG_INFO("Date: {}", inv.date);
    LOG_INFO("================");

    // Print items
    for(const auto& item : inv.items)
    {
        LOG_INFO("{}: ${}", item.name, item.price);
    }

    // Calculate and print total
    double total = 0.0;
    for(const auto& item : inv.items)
    {
        total += item.price;
    }
    LOG_INFO("Total: ${}", total);
}

// AFTER
void
print_header(const invoice& inv)
{
    LOG_INFO("================");
    LOG_INFO("INVOICE #{}", inv.number);
    LOG_INFO("Date: {}", inv.date);
    LOG_INFO("================");
}

void
print_items(const std::vector<line_item>& items)
{
    for(const auto& item : items)
    {
        LOG_INFO("{}: ${}", item.name, item.price);
    }
}

[[nodiscard]] double
calculate_total(const std::vector<line_item>& items)
{
    return std::accumulate(
        items.begin(), items.end(), 0.0,
        [](double sum, const line_item& item) { return sum + item.price; });
}

void
print_invoice(const invoice& inv)
{
    print_header(inv);
    print_items(inv.items);
    LOG_INFO("Total: ${}", calculate_total(inv.items));
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
class supplier
{
public:
    [[nodiscard]] int get_rating() const
    {
        return more_than_five_late_deliveries() ? 2 : 1;
    }

private:
    [[nodiscard]] bool more_than_five_late_deliveries() const
    {
        return m_late_deliveries > k_max_late_deliveries;
    }

    static constexpr int k_max_late_deliveries = 5;
    int                  m_late_deliveries     = 0;
};

// AFTER
class supplier
{
public:
    [[nodiscard]] int get_rating() const
    {
        return m_late_deliveries > k_max_late_deliveries ? 2 : 1;
    }

private:
    static constexpr int k_max_late_deliveries = 5;
    int                  m_late_deliveries     = 0;
};
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
void
handle_resize(const std::string& platform, const std::string& browser, int resize)
{
    if(platform.find("MAC") != std::string::npos &&
       browser.find("IE") != std::string::npos && was_initialized() && resize > 0)
    {
        // ...
    }
}

// AFTER
void
handle_resize(const std::string& platform, const std::string& browser, int resize)
{
    const bool is_mac      = platform.find("MAC") != std::string::npos;
    const bool is_ie       = browser.find("IE") != std::string::npos;
    const bool was_resized = resize > 0;

    if(is_mac && is_ie && was_initialized() && was_resized)
    {
        // ...
    }
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
constexpr double k_bulk_threshold   = 1000.0;
constexpr double k_bulk_discount    = 0.95;
constexpr double k_regular_discount = 0.98;

// BEFORE
class order
{
public:
    [[nodiscard]] double calculate_total() const
    {
        const double base_price = m_quantity * m_item_price;
        if(base_price > k_bulk_threshold)
        {
            return base_price * k_bulk_discount;
        }
        return base_price * k_regular_discount;
    }

private:
    int    m_quantity   = 0;
    double m_item_price = 0.0;
};

// AFTER
class order
{
public:
    [[nodiscard]] double calculate_total() const
    {
        if(base_price() > k_bulk_threshold)
        {
            return base_price() * k_bulk_discount;
        }
        return base_price() * k_regular_discount;
    }

private:
    [[nodiscard]] double base_price() const { return m_quantity * m_item_price; }

    int    m_quantity   = 0;
    double m_item_price = 0.0;
};
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
void
log_shape(double height, double width)
{
    double temp = 2 * (height + width);
    LOG_INFO("Perimeter: {}", temp);
    temp = height * width;
    LOG_INFO("Area: {}", temp);
}

// AFTER
void
log_shape(double height, double width)
{
    const double perimeter = 2 * (height + width);
    LOG_INFO("Perimeter: {}", perimeter);
    const double area = height * width;
    LOG_INFO("Area: {}", area);
}
```

---

#### Remove Assignments to Parameters

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Confusing parameter modification |

**Problem:** Value assigned to a parameter inside method body.
**Solution:** Use a local variable instead of the parameter.

```cpp
constexpr int k_bulk_quantity = 50;

// BEFORE
[[nodiscard]] int
discount(int input_val, int quantity)
{
    if(quantity > k_bulk_quantity)
    {
        input_val -= 2;  // Modifying parameter!
    }
    return input_val;
}

// AFTER
[[nodiscard]] int
discount(int input_val, int quantity)
{
    int result = input_val;
    if(quantity > k_bulk_quantity)
    {
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
class order
{
public:
    [[nodiscard]] double calculate_price() const
    {
        const double primary_base = compute_primary_base(*this);
        const double secondary    = compute_secondary(primary_base);
        const double tertiary     = compute_tertiary(secondary);
        // 100 more lines using all these variables...
        return primary_base + secondary + tertiary;
    }
};

// AFTER
class order;

class price_calculator
{
public:
    explicit price_calculator(const order& source)
    : m_order{ source }
    {}

    [[nodiscard]] double compute()
    {
        m_primary_base = compute_primary_base(m_order);
        m_secondary    = compute_secondary(m_primary_base);
        m_tertiary     = compute_tertiary(m_secondary);
        // Can now Extract Method freely!
        return calculate_final();
    }

private:
    [[nodiscard]] double calculate_final() const
    {
        return m_primary_base + m_secondary + m_tertiary;
    }

    const order& m_order;
    double       m_primary_base = 0.0;
    double       m_secondary    = 0.0;
    double       m_tertiary     = 0.0;
};

class order
{
public:
    [[nodiscard]] double calculate_price() const
    {
        return price_calculator{ *this }.compute();
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
[[nodiscard]] std::string_view
find_person(const std::vector<std::string>& people)
{
    for(const auto& person : people)
    {
        if(person == "Don")
        {
            return "Don";
        }
    }
    for(const auto& person : people)
    {
        if(person == "John")
        {
            return "John";
        }
    }
    for(const auto& person : people)
    {
        if(person == "Kent")
        {
            return "Kent";
        }
    }
    return {};
}

// AFTER
[[nodiscard]] std::string_view
find_person(const std::vector<std::string>& people)
{
    static constexpr std::array<std::string_view, 3> k_candidates = { "Don", "John",
                                                                      "Kent" };
    for(const auto candidate : k_candidates)
    {
        if(std::ranges::find(people, candidate) != people.end())
        {
            return candidate;
        }
    }
    return {};
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
constexpr double k_premium_base_charge  = 10.0;
constexpr int    k_grace_days           = 7;
constexpr double k_premium_daily_charge = 0.85;
constexpr double k_regular_daily_charge = 1.75;

// BEFORE
class account
{
public:
    [[nodiscard]] double overdraft_charge() const
    {
        if(m_type.is_premium())
        {
            double result = k_premium_base_charge;
            if(m_days_overdrawn > k_grace_days)
            {
                result += (m_days_overdrawn - k_grace_days) * k_premium_daily_charge;
            }
            return result;
        }
        return m_days_overdrawn * k_regular_daily_charge;
    }

private:
    account_type m_type;
    int          m_days_overdrawn = 0;
};

// AFTER - method moved to account_type where it belongs
class account_type
{
public:
    [[nodiscard]] bool is_premium() const;

    [[nodiscard]] double overdraft_charge(int days_overdrawn) const
    {
        if(is_premium())
        {
            double result = k_premium_base_charge;
            if(days_overdrawn > k_grace_days)
            {
                result += (days_overdrawn - k_grace_days) * k_premium_daily_charge;
            }
            return result;
        }
        return days_overdrawn * k_regular_daily_charge;
    }
};

class account
{
public:
    [[nodiscard]] double overdraft_charge() const
    {
        return m_type.overdraft_charge(m_days_overdrawn);
    }

private:
    account_type m_type;
    int          m_days_overdrawn = 0;
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
class account
{
    account_type m_type;
    double       m_interest_rate = 0.0;
};

// AFTER - m_interest_rate moved to account_type
class account_type
{
public:
    [[nodiscard]] double interest_rate() const { return m_interest_rate; }
    void                 set_interest_rate(double rate) { m_interest_rate = rate; }

private:
    double m_interest_rate = 0.0;
};

class account
{
public:
    [[nodiscard]] double interest_rate() const { return m_type.interest_rate(); }

private:
    account_type m_type;
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
class person
{
public:
    [[nodiscard]] std::string telephone_number() const
    {
        return "(" + m_office_area_code + ") " + m_office_number;
    }

private:
    std::string m_name;
    std::string m_office_area_code;
    std::string m_office_number;
};

// AFTER
class phone_number
{
public:
    [[nodiscard]] std::string format() const
    {
        return "(" + m_area_code + ") " + m_number;
    }

private:
    std::string m_area_code;
    std::string m_number;
};

class person
{
public:
    [[nodiscard]] std::string telephone_number() const
    {
        return m_office_telephone.format();
    }

private:
    std::string  m_name;
    phone_number m_office_telephone;
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
// BEFORE - phone_number is too thin
class phone_number
{
public:
    [[nodiscard]] std::string number() const { return m_number; }

private:
    std::string m_number;
};

class person
{
public:
    [[nodiscard]] std::string telephone_number() const { return m_phone.number(); }

private:
    phone_number m_phone;
};

// AFTER
class person
{
public:
    [[nodiscard]] std::string telephone_number() const { return m_telephone_number; }

private:
    std::string m_telephone_number;
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
// BEFORE - client knows about department
class person
{
public:
    [[nodiscard]] department* get_department() const { return m_department; }

private:
    department* m_department = nullptr;
};
// Client: auto* manager = john.get_department()->manager();

// AFTER - department hidden
class person
{
public:
    [[nodiscard]] person* manager() const { return m_department->manager(); }

private:
    department* m_department = nullptr;
};
// Client: auto* manager = john.manager();
```

---

#### Remove Middle Man

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Middle Man |

**Problem:** Class has too many methods that simply delegate.
**Solution:** Delete delegation methods; have clients call the delegate directly.

```cpp
// BEFORE - person delegates everything
class person
{
public:
    [[nodiscard]] person* manager() const { return m_department->manager(); }
    [[nodiscard]] double  budget() const { return m_department->budget(); }
    [[nodiscard]] int     team_size() const { return m_department->team_size(); }
    // ... 10 more delegation methods

private:
    department* m_department = nullptr;
};

// AFTER - expose delegate
class person
{
public:
    [[nodiscard]] department* get_department() const { return m_department; }

private:
    department* m_department = nullptr;
};
// Client: auto budget = john.get_department()->budget();
```

---

#### Introduce Foreign Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Low | Low | Incomplete Library Class (1-2 methods) |

**Problem:** Utility class lacks a method you need, and you cannot modify it.
**Solution:** Create the method in your client class, taking the utility object as parameter.

```cpp
// Cannot modify the date class
class report
{
    // Foreign method - ideally should be in date
    [[nodiscard]] static date next_day(const date& current)
    {
        return date{ current.year(), current.month(), current.day() + 1 };
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
constexpr int k_sunday   = 0;
constexpr int k_saturday = 6;

// Subclass approach
class extended_date : public date
{
public:
    using date::date;

    [[nodiscard]] date next_day() const { return date{ year(), month(), day() + 1 }; }
    [[nodiscard]] bool is_weekend() const
    {
        const int weekday = day_of_week();
        return weekday == k_sunday || weekday == k_saturday;
    }
};

// Wrapper approach (when inheritance is not possible)
class date_wrapper
{
public:
    explicit date_wrapper(const date& source)
    : m_date{ source }
    {}

    [[nodiscard]] int  year() const { return m_date.year(); }
    [[nodiscard]] int  month() const { return m_date.month(); }
    [[nodiscard]] int  day() const { return m_date.day(); }
    [[nodiscard]] date next_day() const { return date{ year(), month(), day() + 1 }; }

private:
    date m_date;
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
class int_range
{
public:
    [[nodiscard]] bool includes(int arg) const
    {
        return arg >= m_low && arg <= m_high;  // Direct access
    }

private:
    int m_low  = 0;
    int m_high = 0;
};

// AFTER
class int_range
{
public:
    [[nodiscard]] int  low() const { return m_low; }
    [[nodiscard]] int  high() const { return m_high; }
    [[nodiscard]] bool includes(int arg) const
    {
        return arg >= low() && arg <= high();  // Via getters
    }

private:
    int m_low  = 0;
    int m_high = 0;
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
class order
{
    std::string m_customer_name;
    std::string m_customer_email;
    std::string m_customer_phone;
};

// AFTER
class customer
{
public:
    [[nodiscard]] bool        is_valid_email() const;
    [[nodiscard]] std::string formatted_phone() const;

private:
    std::string m_name;
    std::string m_email;
    std::string m_phone;
};

class order
{
    customer m_customer;
};
```

---

#### Replace Magic Number with Symbolic Constant

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Magic Numbers |

```cpp
// BEFORE
[[nodiscard]] double
potential_energy(double mass, double height)
{
    return mass * 9.81 * height;
}

// AFTER
constexpr double k_gravitational_acceleration = 9.81;  // m/s^2

[[nodiscard]] double
potential_energy(double mass, double height)
{
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
class person
{
public:
    std::string name;  // Public field!
};

// AFTER
class person
{
public:
    [[nodiscard]] const std::string& name() const { return m_name; }
    void set_name(const std::string& value) { m_name = value; }

private:
    std::string m_name;
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
class person
{
public:
    std::vector<course>& courses() { return m_courses; }  // Dangerous!

private:
    std::vector<course> m_courses;
};

// AFTER
class person
{
public:
    [[nodiscard]] const std::vector<course>& courses() const { return m_courses; }
    void add_course(const course& entry) { m_courses.push_back(entry); }
    void remove_course(const course& entry) { std::erase(m_courses, entry); }

private:
    std::vector<course> m_courses;
};
```

---

#### Replace Type Code with Class

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Primitive Obsession (type codes without behavior) |

```cpp
// BEFORE
class person
{
    static constexpr int k_o  = 0;
    static constexpr int k_a  = 1;
    static constexpr int k_b  = 2;
    static constexpr int k_ab = 3;

    int m_blood_type = k_o;
};

// AFTER
class blood_type
{
public:
    static const blood_type k_o;
    static const blood_type k_a;
    static const blood_type k_b;
    static const blood_type k_ab;

    bool operator==(const blood_type& other) const = default;

private:
    explicit blood_type(int code)
    : m_code{ code }
    {}

    int m_code = 0;
};

class person
{
    blood_type m_blood_type = blood_type::k_o;
};
```

---

#### Replace Type Code with Subclasses

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | High | Switch Statements, Primitive Obsession (type codes with behavior) |

```cpp
constexpr int k_engineer_bonus = 100;
constexpr int k_manager_bonus  = 500;
constexpr int k_salesman_bonus = 200;

// BEFORE
enum class employee_kind
{
    engineer,
    manager,
    salesman
};

class employee
{
public:
    [[nodiscard]] int bonus() const
    {
        switch(m_kind)
        {
            case employee_kind::engineer: return k_engineer_bonus;
            case employee_kind::manager: return k_manager_bonus;
            case employee_kind::salesman: return k_salesman_bonus;
            default:
                throw std::logic_error(fmt::format("employee_kind {} not handled",
                                                   static_cast<int>(m_kind)));
        }
    }

private:
    employee_kind m_kind = employee_kind::engineer;
};

// AFTER
class employee_interface
{
public:
    virtual ~employee_interface()           = default;
    [[nodiscard]] virtual int bonus() const = 0;
};

class engineer final : public employee_interface
{
public:
    [[nodiscard]] int bonus() const override { return k_engineer_bonus; }
};

class manager final : public employee_interface
{
public:
    [[nodiscard]] int bonus() const override { return k_manager_bonus; }
};

class salesman final : public employee_interface
{
public:
    [[nodiscard]] int bonus() const override { return k_salesman_bonus; }
};
```

---

#### Replace Type Code with State/Strategy

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | High | Switch Statements when subclassing is impossible (type changes at runtime) |

```cpp
// BEFORE
enum class employee_kind
{
    engineer,
    salesman,
    manager
};

class employee
{
public:
    [[nodiscard]] int pay_amount() const
    {
        switch(m_kind)
        {
            case employee_kind::engineer: return m_monthly_salary;
            case employee_kind::salesman: return m_monthly_salary + m_commission;
            case employee_kind::manager: return m_monthly_salary + m_bonus;
            default:
                throw std::logic_error(fmt::format("employee_kind {} not handled",
                                                   static_cast<int>(m_kind)));
        }
    }

private:
    employee_kind m_kind           = employee_kind::engineer;
    int           m_monthly_salary = 0;
    int           m_commission     = 0;
    int           m_bonus          = 0;
};

// AFTER
class employee;

class employee_type_interface
{
public:
    virtual ~employee_type_interface()                              = default;
    [[nodiscard]] virtual int pay_amount(const employee& emp) const = 0;
};

class employee
{
public:
    [[nodiscard]] int monthly_salary() const { return m_monthly_salary; }

    void set_type(std::unique_ptr<employee_type_interface> next)
    {
        m_type = std::move(next);  // Can change at runtime
    }
    [[nodiscard]] int pay_amount() const { return m_type->pay_amount(*this); }

private:
    std::unique_ptr<employee_type_interface> m_type;
    int                                      m_monthly_salary = 0;
};

class engineer final : public employee_type_interface
{
public:
    [[nodiscard]] int pay_amount(const employee& emp) const override
    {
        return emp.monthly_salary();
    }
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
constexpr int k_summer_start = 152;  // day of year
constexpr int k_summer_end   = 243;

// BEFORE
class tariff
{
public:
    [[nodiscard]] double calculate_charge(int day_of_year, int quantity) const
    {
        if(day_of_year < k_summer_start || day_of_year > k_summer_end)
        {
            return quantity * m_winter_rate + m_winter_service_charge;
        }
        return quantity * m_summer_rate;
    }

private:
    double m_winter_rate           = 0.0;
    double m_winter_service_charge = 0.0;
    double m_summer_rate           = 0.0;
};

// AFTER
class tariff
{
public:
    [[nodiscard]] double calculate_charge(int day_of_year, int quantity) const
    {
        return is_summer(day_of_year) ? summer_charge(quantity) : winter_charge(quantity);
    }

private:
    [[nodiscard]] static bool is_summer(int day_of_year)
    {
        return day_of_year >= k_summer_start && day_of_year <= k_summer_end;
    }
    [[nodiscard]] double summer_charge(int quantity) const
    {
        return quantity * m_summer_rate;
    }
    [[nodiscard]] double winter_charge(int quantity) const
    {
        return quantity * m_winter_rate + m_winter_service_charge;
    }

    double m_winter_rate           = 0.0;
    double m_winter_service_charge = 0.0;
    double m_summer_rate           = 0.0;
};
```

---

#### Consolidate Conditional Expression

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Duplicate conditionals |

**Problem:** Multiple conditionals leading to the same result.
**Solution:** Combine into a single expression and extract to a method.

```cpp
constexpr int k_max_months_disabled = 12;

// BEFORE
class employee
{
public:
    [[nodiscard]] double disability_amount() const
    {
        if(m_seniority < 2)
        {
            return 0.0;
        }
        if(m_months_disabled > k_max_months_disabled)
        {
            return 0.0;
        }
        if(m_is_part_time)
        {
            return 0.0;
        }
        return calculate_disability();
    }

private:
    [[nodiscard]] double calculate_disability() const;

    int  m_seniority       = 0;
    int  m_months_disabled = 0;
    bool m_is_part_time    = false;
};

// AFTER
class employee
{
public:
    [[nodiscard]] double disability_amount() const
    {
        if(is_not_eligible_for_disability())
        {
            return 0.0;
        }
        return calculate_disability();
    }

private:
    [[nodiscard]] bool is_not_eligible_for_disability() const
    {
        return m_seniority < 2 || m_months_disabled > k_max_months_disabled ||
               m_is_part_time;
    }
    [[nodiscard]] double calculate_disability() const;

    int  m_seniority       = 0;
    int  m_months_disabled = 0;
    bool m_is_part_time    = false;
};
```

---

#### Consolidate Duplicate Conditional Fragments

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Duplicate Code in branches |

**Problem:** Same code in all branches of a conditional.
**Solution:** Move it outside the conditional.

```cpp
constexpr double k_special_discount = 0.95;
constexpr double k_regular_discount = 0.98;

// BEFORE
[[nodiscard]] double
checkout(double price)
{
    double total = 0.0;
    if(is_special_deal())
    {
        total = price * k_special_discount;
        send_notification();  // Duplicated!
    }
    else
    {
        total = price * k_regular_discount;
        send_notification();  // Duplicated!
    }
    return total;
}

// AFTER
[[nodiscard]] double
checkout(double price)
{
    const double total =
        price * (is_special_deal() ? k_special_discount : k_regular_discount);
    send_notification();  // Moved outside
    return total;
}
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
class payroll
{
public:
    [[nodiscard]] double get_pay_amount() const
    {
        double result = 0.0;
        if(m_is_dead)
        {
            result = dead_amount();
        }
        else
        {
            if(m_is_separated)
            {
                result = separated_amount();
            }
            else
            {
                if(m_is_retired)
                {
                    result = retired_amount();
                }
                else
                {
                    result = normal_pay_amount();
                }
            }
        }
        return result;
    }

private:
    [[nodiscard]] double dead_amount() const;
    [[nodiscard]] double separated_amount() const;
    [[nodiscard]] double retired_amount() const;
    [[nodiscard]] double normal_pay_amount() const;

    bool m_is_dead      = false;
    bool m_is_separated = false;
    bool m_is_retired   = false;
};

// AFTER
class payroll
{
public:
    [[nodiscard]] double get_pay_amount() const
    {
        if(m_is_dead)
        {
            return dead_amount();
        }
        if(m_is_separated)
        {
            return separated_amount();
        }
        if(m_is_retired)
        {
            return retired_amount();
        }
        return normal_pay_amount();
    }

private:
    [[nodiscard]] double dead_amount() const;
    [[nodiscard]] double separated_amount() const;
    [[nodiscard]] double retired_amount() const;
    [[nodiscard]] double normal_pay_amount() const;

    bool m_is_dead      = false;
    bool m_is_separated = false;
    bool m_is_retired   = false;
};
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
enum class bird_kind
{
    european,
    african,
    norwegian_blue
};

class bird
{
public:
    [[nodiscard]] double get_speed() const
    {
        switch(m_kind)
        {
            case bird_kind::european: return base_speed();
            case bird_kind::african: return base_speed() - m_load_factor * m_num_coconuts;
            case bird_kind::norwegian_blue:
                return m_is_nailed ? 0.0 : base_speed(m_voltage);
            default:
                throw std::logic_error(
                    fmt::format("bird_kind {} not handled", static_cast<int>(m_kind)));
        }
    }

private:
    bird_kind m_kind         = bird_kind::european;
    double    m_load_factor  = 0.0;
    int       m_num_coconuts = 0;
    bool      m_is_nailed    = false;
    double    m_voltage      = 0.0;
};

// AFTER
class bird_interface
{
public:
    virtual ~bird_interface()                      = default;
    [[nodiscard]] virtual double get_speed() const = 0;
};

class european final : public bird_interface
{
public:
    [[nodiscard]] double get_speed() const override { return base_speed(); }
};

class african final : public bird_interface
{
public:
    [[nodiscard]] double get_speed() const override
    {
        return base_speed() - m_load_factor * m_num_coconuts;
    }

private:
    double m_load_factor  = 0.0;
    int    m_num_coconuts = 0;
};

class norwegian_blue final : public bird_interface
{
public:
    [[nodiscard]] double get_speed() const override
    {
        return m_is_nailed ? 0.0 : base_speed(m_voltage);
    }

private:
    bool   m_is_nailed = false;
    double m_voltage   = 0.0;
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
// BEFORE - scattered null checks
[[nodiscard]] billing_plan
plan_for(const customer* buyer)
{
    if(buyer != nullptr)
    {
        return buyer->plan();
    }
    return billing_plan::basic();
}

// AFTER
class null_customer final : public customer
{
public:
    [[nodiscard]] bool         is_null() const override { return true; }
    [[nodiscard]] billing_plan plan() const override { return billing_plan::basic(); }
};

// Usage, no null checks: the lookup returns a null_customer instead of nullptr
[[nodiscard]] billing_plan
plan_for(const customer& buyer)
{
    return buyer.plan();
}
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
class employee
{
public:
    [[nodiscard]] double get_expense_limit() const
    {
        return m_expense_limit != k_null_expense
                   ? m_expense_limit
                   : m_primary_project->member_expense_limit();
    }

private:
    static constexpr double k_null_expense = -1.0;

    double         m_expense_limit   = k_null_expense;
    const project* m_primary_project = nullptr;
};

// AFTER
class employee
{
public:
    [[nodiscard]] double get_expense_limit() const
    {
        assert(m_expense_limit != k_null_expense || m_primary_project != nullptr);
        return m_expense_limit != k_null_expense
                   ? m_expense_limit
                   : m_primary_project->member_expense_limit();
    }

private:
    static constexpr double k_null_expense = -1.0;

    double         m_expense_limit   = k_null_expense;
    const project* m_primary_project = nullptr;
};
```

---

### Category E: Simplifying Method Calls

#### Rename Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Low | Unclear intent, Comments |

```cpp
// BEFORE
class person
{
public:
    [[nodiscard]] std::string get_tlp() const { return m_telephone; }

private:
    std::string m_telephone;
};

// AFTER
class person
{
public:
    [[nodiscard]] std::string get_telephone_number() const { return m_telephone; }

private:
    std::string m_telephone;
};
```

---

#### Add Parameter / Remove Parameter

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Medium | Missing/unused data |

```cpp
// Remove Parameter
// BEFORE
class counter
{
public:
    void set_value(int value, int unused) { m_value = value; }

private:
    int m_value = 0;
};

// AFTER
class counter
{
public:
    void set_value(int value) { m_value = value; }

private:
    int m_value = 0;
};
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
class cart
{
public:
    [[nodiscard]] int get_total_and_clear()
    {
        const int result = calculate_total();
        clear_items();  // Side effect!
        return result;
    }

private:
    [[nodiscard]] int calculate_total() const;
    void              clear_items();
};

// AFTER
class cart
{
public:
    [[nodiscard]] int get_total() const { return calculate_total(); }  // Query
    void              clear() { clear_items(); }                       // Modifier

private:
    [[nodiscard]] int calculate_total() const;
    void              clear_items();
};
```

---

#### Parameterize Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Similar methods, Duplicate Code |

```cpp
// BEFORE
class employee
{
public:
    void five_percent_raise() { m_salary *= k_five_percent_factor; }
    void ten_percent_raise() { m_salary *= k_ten_percent_factor; }

private:
    static constexpr double k_five_percent_factor = 1.05;
    static constexpr double k_ten_percent_factor  = 1.10;

    double m_salary = 0.0;
};

// AFTER
class employee
{
public:
    void raise(double rate) { m_salary *= (1 + rate); }

private:
    double m_salary = 0.0;
};
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
[[nodiscard]] double
amount_invoiced(const date& start, const date& end);
[[nodiscard]] double
amount_received(const date& start, const date& end);
[[nodiscard]] double
amount_overdue(const date& start, const date& end);

// AFTER
class date_range
{
public:
    date_range(const date& start, const date& end)
    : m_start{ start }
    , m_end{ end }
    {}

    [[nodiscard]] bool includes(const date& when) const
    {
        return when >= m_start && when <= m_end;
    }

private:
    date m_start;
    date m_end;
};

[[nodiscard]] double
amount_invoiced(const date_range& range);
[[nodiscard]] double
amount_received(const date_range& range);
[[nodiscard]] double
amount_overdue(const date_range& range);
```

---

#### Preserve Whole Object

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Long Parameter List, Feature Envy |

```cpp
// BEFORE
[[nodiscard]] bool
is_within_plan(const heating_plan& plan, const temp_range& days_temp_range)
{
    const int low  = days_temp_range.low();
    const int high = days_temp_range.high();
    return plan.within_range(low, high);
}

// AFTER
[[nodiscard]] bool
is_within_plan(const heating_plan& plan, const temp_range& days_temp_range)
{
    return plan.within_range(days_temp_range);
}
```

---

#### Replace Constructor with Factory Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Complex construction, Type-based creation |

```cpp
// AFTER
enum class employee_kind
{
    engineer,
    manager,
    salesman
};

class employee
{
public:
    virtual ~employee() = default;

    [[nodiscard]] static std::unique_ptr<employee> create(employee_kind kind);

protected:
    employee() = default;
};

class engineer final : public employee
{};

class manager final : public employee
{};

class salesman final : public employee
{};

std::unique_ptr<employee>
employee::create(employee_kind kind)
{
    switch(kind)
    {
        case employee_kind::engineer: return std::make_unique<engineer>();
        case employee_kind::manager: return std::make_unique<manager>();
        case employee_kind::salesman: return std::make_unique<salesman>();
        default:
            throw std::logic_error(
                fmt::format("employee_kind {} not handled", static_cast<int>(kind)));
    }
}
```

---

#### Replace Error Code with Exception

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Error handling clutter |

```cpp
// BEFORE
class account
{
public:
    [[nodiscard]] int withdraw(int amount)
    {
        if(amount > m_balance)
        {
            return -1;  // Error code
        }
        m_balance -= amount;
        return 0;
    }

private:
    int m_balance = 0;
};

// AFTER
class account
{
public:
    void withdraw(int amount)
    {
        if(amount > m_balance)
        {
            throw std::runtime_error("Insufficient funds");
        }
        m_balance -= amount;
    }

private:
    int m_balance = 0;
};
```

---

#### Replace Exception with Test

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Exception misuse |

```cpp
// BEFORE
class ledger
{
public:
    [[nodiscard]] double value_for_period(std::size_t period) const
    {
        try
        {
            return m_values.at(period);
        } catch(const std::out_of_range&)
        {
            return 0.0;
        }
    }

private:
    std::vector<double> m_values;
};

// AFTER
class ledger
{
public:
    [[nodiscard]] double value_for_period(std::size_t period) const
    {
        if(period >= m_values.size())
        {
            return 0.0;
        }
        return m_values[period];
    }

private:
    std::vector<double> m_values;
};
```

---

### Category F: Dealing with Generalization

#### Pull Up Field / Pull Up Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Duplicate Code in subclasses |

```cpp
// BEFORE
class salesman : public employee
{
    std::string m_name;  // Duplicated
};
class engineer : public employee
{
    std::string m_name;  // Duplicated
};

// AFTER
class employee
{
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
// BEFORE - duplicate init in each subclass
class manager : public employee
{
public:
    manager(std::string name, std::string employee_id, int grade)
    {
        m_name        = std::move(name);
        m_employee_id = std::move(employee_id);
        m_grade       = grade;
    }

private:
    int m_grade = 0;
};

// AFTER
class employee
{
protected:
    employee(std::string name, std::string employee_id)
    : m_name{ std::move(name) }
    , m_employee_id{ std::move(employee_id) }
    {}

    std::string m_name;
    std::string m_employee_id;
};

class manager : public employee
{
public:
    manager(std::string name, std::string employee_id, int grade)
    : employee{ std::move(name), std::move(employee_id) }
    , m_grade{ grade }
    {}

private:
    int m_grade = 0;
};
```

---

#### Push Down Method / Push Down Field

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Superclass has irrelevant members |

```cpp
// BEFORE - get_quota only relevant to salesman
class employee
{
public:
    virtual ~employee() = default;
    [[nodiscard]] virtual double get_quota() const { return 0.0; }  // Not relevant to all
};

// AFTER
class employee
{
public:
    virtual ~employee() = default;
};

class salesman : public employee
{
public:
    [[nodiscard]] double get_quota() const { return m_quota; }  // Moved here

private:
    double m_quota = 0.0;
};
```

---

#### Extract Subclass

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Large Class, features used only sometimes |

```cpp
// BEFORE
class job_item
{
public:
    [[nodiscard]] double unit_price() const
    {
        return m_is_labor ? m_employee->rate() : m_unit_price;
    }

private:
    double          m_unit_price = 0.0;
    const employee* m_employee   = nullptr;  // Only for labor items
    bool            m_is_labor   = false;
};

// AFTER
class job_item_interface
{
public:
    virtual ~job_item_interface()                   = default;
    [[nodiscard]] virtual double unit_price() const = 0;
};

class parts_item final : public job_item_interface
{
public:
    [[nodiscard]] double unit_price() const override { return m_unit_price; }

private:
    double m_unit_price = 0.0;
};

class labor_item final : public job_item_interface
{
public:
    [[nodiscard]] double unit_price() const override { return m_employee->rate(); }

private:
    const employee* m_employee = nullptr;
};
```

---

#### Extract Superclass

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Duplicate Code, similar classes |

```cpp
// BEFORE - department and employee both have m_name and an annual cost
class department
{
    std::string m_name;
    /* ... */
};
class employee
{
    std::string m_name;
    /* ... */
};

// AFTER
class party_interface
{
public:
    virtual ~party_interface()                       = default;
    [[nodiscard]] virtual double annual_cost() const = 0;

protected:
    std::string m_name;
};

class department final : public party_interface
{
public:
    [[nodiscard]] double annual_cost() const override;
};

class employee final : public party_interface
{
public:
    [[nodiscard]] double annual_cost() const override;
};
```

---

#### Extract Interface

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Tight coupling, untestable code |

```cpp
class billable_interface
{
public:
    virtual ~billable_interface()                          = default;
    [[nodiscard]] virtual double rate() const              = 0;
    [[nodiscard]] virtual bool   has_special_skill() const = 0;
};

class employee final : public billable_interface
{
public:
    [[nodiscard]] double rate() const override;
    [[nodiscard]] bool   has_special_skill() const override;
};
// Client depends only on billable_interface -- can mock for testing
```

---

#### Collapse Hierarchy

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Low | Lazy Class, unnecessary subclass |

```cpp
// BEFORE
class employee
{
public:
    virtual ~employee() = default;
    [[nodiscard]] virtual double rate() const { return m_base_rate; }

private:
    double m_base_rate = 0.0;
};
class salesman : public employee
{
    // adds nothing
};

// AFTER - salesman class deleted
class employee
{
public:
    [[nodiscard]] double rate() const { return m_base_rate; }

private:
    double m_base_rate = 0.0;
};
```

---

#### Form Template Method

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Similar algorithms in subclasses |

```cpp
constexpr double k_tax_rate            = 0.2;
constexpr double k_lifeline_base_share = 0.5;
constexpr double k_lifeline_tax_share  = 0.2;

// AFTER -- Template Method pattern
class site_interface
{
public:
    virtual ~site_interface() = default;

    [[nodiscard]] double bill_amount() const  // Template method
    {
        return get_base_amount() + get_tax_amount();
    }

protected:
    [[nodiscard]] virtual double get_base_amount() const = 0;
    [[nodiscard]] virtual double get_tax_amount() const  = 0;
};

class residential_site final : public site_interface
{
protected:
    [[nodiscard]] double get_base_amount() const override { return m_units * m_rate; }
    [[nodiscard]] double get_tax_amount() const override
    {
        return get_base_amount() * k_tax_rate;
    }

private:
    double m_units = 0.0;
    double m_rate  = 0.0;
};

class lifeline_site final : public site_interface
{
protected:
    [[nodiscard]] double get_base_amount() const override
    {
        return m_units * m_rate * k_lifeline_base_share;
    }
    [[nodiscard]] double get_tax_amount() const override
    {
        return get_base_amount() * k_tax_rate * k_lifeline_tax_share;
    }

private:
    double m_units = 0.0;
    double m_rate  = 0.0;
};
```

---

#### Replace Inheritance with Delegation

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| High | Medium | Refused Bequest, wrong hierarchy |

```cpp
// BEFORE - stack should not inherit from std::vector
class stack : public std::vector<int>
{
    // Exposes ALL vector methods!
};

// AFTER - composition
class stack
{
public:
    void push(int element) { m_data.push_back(element); }
    int  pop()
    {
        const int result = m_data.back();
        m_data.pop_back();
        return result;
    }
    [[nodiscard]] bool empty() const { return m_data.empty(); }
    // Only exposes stack operations

private:
    std::vector<int> m_data;
};
```

---

#### Replace Delegation with Inheritance

| Impact | Risk | Fixes |
| -------- | ------ | ------- |
| Medium | Medium | Excessive delegation (opposite of above) |

```cpp
// BEFORE - too much delegation
class employee
{
public:
    [[nodiscard]] std::string name() const { return m_person.name(); }
    void set_name(std::string value) { m_person.set_name(std::move(value)); }
    [[nodiscard]] std::string address() const { return m_person.address(); }
    // ... 10 more delegation methods

private:
    person m_person;
};

// AFTER
class employee : public person
{
    // Inherits all person methods directly
};
```

---

### References

- [Refactoring.Guru - Code Smells](https://refactoring.guru/refactoring/smells)
- [Refactoring.Guru - Refactoring Techniques](https://refactoring.guru/refactoring/techniques)
- Martin Fowler, *Refactoring: Improving the Design of Existing Code*
