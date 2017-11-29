#ifndef _SRC_XML_EXPR_H
#define _SRC_XML_EXPR_H

#include <exception>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <string.h>

namespace xml {
class exception_t : public std::exception {
 public:
  explicit exception_t(const std::string& msg) : str_(msg) {}
  const char* what() const throw() { return str_.c_str(); }

 protected:
  const std::string str_;
};

typedef uint64_t args_t;
typedef std::map<std::string, args_t> args_map_t;
class Expr;

template <class T> class any_cache_t {
 public:
  virtual ~any_cache_t() {}
  virtual bool Lookup(const std::string& name, T& result) const = 0;
};

typedef any_cache_t<std::string> expr_cache_t;
typedef any_cache_t<args_t> args_cache_t;

class bin_expr_t {
 public:
  static const bin_expr_t* CreateExpr(const bin_expr_t* arg1, const bin_expr_t* arg2,
                                      const char op);
  static const bin_expr_t* CreateArg(Expr* obj, const std::string str);

  bin_expr_t() : arg1_(NULL), arg2_(NULL) {}
  bin_expr_t(const bin_expr_t* arg1, const bin_expr_t* arg2) : arg1_(arg1), arg2_(arg2) {}

  virtual args_t Eval(const args_cache_t& args) const = 0;
  virtual std::string Symbol() const = 0;

  std::string String() const {
    std::string str;
    if (arg1_) {
      str = "(" + arg1_->String() + " " + Symbol() + " " + arg2_->String() + ")";
    } else
      str = Symbol();
    return str;
  }

 protected:
  const bin_expr_t* arg1_;
  const bin_expr_t* arg2_;
};

class Expr {
 public:
  explicit Expr(const std::string& expr, const expr_cache_t* cache)
      : expr_(expr), pos_(0), sub_count_(0), cache_(cache) {
    sub_vec_ = new std::vector<const Expr*>;
    var_vec_ = new std::vector<std::string>;
    tree_ = ParseExpr();
  }

  explicit Expr(const std::string& expr, const Expr* obj)
      : expr_(expr),
        pos_(0),
        sub_count_(0),
        cache_(obj->cache_),
        sub_vec_(obj->sub_vec_),
        var_vec_(obj->var_vec_) {
    sub_vec_->push_back(this);
    tree_ = ParseExpr();
    if (!SubCheck()) throw exception_t("expr '" + expr_ + "', bad parenthesis count");
  }

  ~Expr() {
    delete cache_;
    for (auto it : *sub_vec_) delete it;
    delete sub_vec_;
    delete var_vec_;
  }

  std::string GetStr() const { return expr_; }
  const expr_cache_t* GetCache() const { return cache_; }
  const bin_expr_t* GetTree() const { return tree_; }

  args_t Eval(const args_cache_t& args) const { return tree_->Eval(args); }

  std::string Lookup(const std::string& str) const {
    std::string result;
    if (cache_ && !(cache_->Lookup(str, result)))
      throw exception_t("expr '" + expr_ + "', lookup '" + str + "' failed");
    return result;
  }

  void AddVar(const std::string& str) {
    bool found = false;
    for (std::string s : *var_vec_)
      if (s == str) found = true;
    if (!found) var_vec_->push_back(str);
  }

  const std::vector<std::string>& GetVars() const { return *var_vec_; }

  std::string String() const { return tree_->String(); }

 private:
  const bin_expr_t* ParseExpr() {
    const bin_expr_t* expr = ParseArg();
    while (!IsEnd()) {
      const char op = Symb();
      const bin_expr_t* second_arg = NULL;
      if (IsSymb(')')) {
        Next();
        SubClose();
        break;
      }
      if (IsSymb('*') || IsSymb('/')) {
        Next();
        second_arg = ParseArg();
        expr = bin_expr_t::CreateExpr(expr, second_arg, op);
      } else if (IsSymb('+') || IsSymb('-')) {
        Next();
        second_arg = ParseExpr();
        expr = bin_expr_t::CreateExpr(expr, second_arg, op);
        break;
      } else {
        throw exception_t("expr '" + expr_ + "', bad operator '" + op + "'");
      }
    }
    return expr;
  }

  const bin_expr_t* ParseArg() {
    const bin_expr_t* arg = NULL;
    if (IsSymb('(')) {
      Next();
      SubOpen();
      arg = ParseExpr();
    } else {
      const unsigned pos = FindOp();
      const std::string str = CutTill(pos);
      arg = bin_expr_t::CreateArg(this, str);
      if (arg == NULL) throw exception_t("expr '" + expr_ + "', bad argument '" + str + "'");
    }
    return arg;
  }

  char Symb() const { return Symb(pos_); }
  char Symb(const unsigned ind) const { return expr_[ind]; }
  bool IsEnd() const { return (pos_ >= expr_.length()); }
  bool IsSymb(const char c) const { return IsSymb(pos_, c); }
  bool IsSymb(const unsigned ind, const char c) const { return (expr_[ind] == c); }
  void Next() { ++pos_; }
  void SubOpen() { ++sub_count_; }
  void SubClose() { --sub_count_; }
  bool SubCheck() const { return (sub_count_ == 0); }
  unsigned FindOp() const {
    unsigned i = pos_;
    while (i < expr_.length()) {
      switch (Symb(i)) {
        case '*':
        case '/':
        case '+':
        case '-':
        case '(':
        case ')':
          goto end;
      }
      ++i;
    }
  end:
    return i;
  }
  std::string CutTill(const unsigned pos) {
    const std::string str = (pos > pos_) ? expr_.substr(pos_, pos - pos_) : "";
    pos_ = pos;
    return str;
  }

  const std::string expr_;
  unsigned pos_;
  unsigned sub_count_;
  const bin_expr_t* tree_;
  const expr_cache_t* const cache_;
  std::vector<const Expr*>* sub_vec_;
  std::vector<std::string>* var_vec_;
};

class add_expr_t : public bin_expr_t {
 public:
  add_expr_t(const bin_expr_t* arg1, const bin_expr_t* arg2) : bin_expr_t(arg1, arg2) {}
  args_t Eval(const args_cache_t& args) const { return (arg1_->Eval(args) + arg2_->Eval(args)); }
  std::string Symbol() const { return "+"; }
};
class sub_expr_t : public bin_expr_t {
 public:
  sub_expr_t(const bin_expr_t* arg1, const bin_expr_t* arg2) : bin_expr_t(arg1, arg2) {}
  args_t Eval(const args_cache_t& args) const { return (arg1_->Eval(args) - arg2_->Eval(args)); }
  std::string Symbol() const { return "-"; }
};
class mul_expr_t : public bin_expr_t {
 public:
  mul_expr_t(const bin_expr_t* arg1, const bin_expr_t* arg2) : bin_expr_t(arg1, arg2) {}
  args_t Eval(const args_cache_t& args) const { return (arg1_->Eval(args) * arg2_->Eval(args)); }
  std::string Symbol() const { return "*"; }
};
class div_expr_t : public bin_expr_t {
 public:
  div_expr_t(const bin_expr_t* arg1, const bin_expr_t* arg2) : bin_expr_t(arg1, arg2) {}
  args_t Eval(const args_cache_t& args) const { return (arg1_->Eval(args) / arg2_->Eval(args)); }
  std::string Symbol() const { return "/"; }
};
class const_expr_t : public bin_expr_t {
 public:
  const_expr_t(const args_t value) : value_(value) {}
  args_t Eval(const args_cache_t&) const { return value_; }
  std::string Symbol() const {
    std::ostringstream os;
    os << value_;
    return os.str();
  }

 private:
  const args_t value_;
};
class var_expr_t : public bin_expr_t {
 public:
  var_expr_t(const std::string name) : name_(name) {}
  args_t Eval(const args_cache_t& args) const {
    args_t result = 0;
    if (!args.Lookup(name_, result)) throw exception_t("expr arg lookup '" + name_ + "' failed");
    return result;
  }
  std::string Symbol() const { return name_; }

 private:
  const std::string name_;
};

inline const bin_expr_t* bin_expr_t::CreateExpr(const bin_expr_t* arg1, const bin_expr_t* arg2,
                                                const char op) {
  const bin_expr_t* expr = NULL;
  switch (op) {
    case '+':
      expr = new add_expr_t(arg1, arg2);
      break;
    case '-':
      expr = new sub_expr_t(arg1, arg2);
      break;
    case '*':
      expr = new mul_expr_t(arg1, arg2);
      break;
    case '/':
      expr = new div_expr_t(arg1, arg2);
      break;
  }
  return expr;
}

inline const bin_expr_t* bin_expr_t::CreateArg(Expr* obj, const std::string str) {
  const bin_expr_t* arg = NULL;
  const unsigned i = strspn(str.c_str(), "1234567890");
  if (i == str.length()) {
    const unsigned value = atoi(str.c_str());
    arg = new const_expr_t(value);
  } else {
    const std::string sub_expr = obj->Lookup(str);
    if (sub_expr.empty()) {
      arg = new var_expr_t(str);
      obj->AddVar(str);
    } else {
      const Expr* expr = new Expr(sub_expr, obj);
      arg = expr->GetTree();
    }
  }
  return arg;
}

}  // namespace xml

#endif  // _SRC_XML_EXPR_H
