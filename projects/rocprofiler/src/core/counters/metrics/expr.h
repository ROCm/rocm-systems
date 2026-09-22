/******************************************************************************
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#ifndef _SRC_CORE_COUNTERS_METRICS_XML_EXPR_H
#define _SRC_CORE_COUNTERS_METRICS_XML_EXPR_H

#include <exception>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <string.h>
#include <float.h>

namespace xml {
class exception_t : public std::exception {
 public:
  explicit exception_t(const std::string& msg) : str_(msg) {}
  const char* what() const throw() { return str_.c_str(); }

 protected:
  const std::string str_;
};

class div_zero_exception_t : public exception_t {
 public:
  explicit div_zero_exception_t(const std::string& msg)
      : exception_t("Divide by zero exception " + msg) {}
};

typedef double args_t;
static const args_t ARGS_MAX = DBL_MAX;
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
  virtual ~bin_expr_t() {
    if (arg1_) delete arg1_;
    if (arg2_) delete arg2_;
  }

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
      : expr_(expr), pos_(0), sub_count_(0), cache_(cache), is_sub_expr_(false) {
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
        var_vec_(obj->var_vec_),
        is_sub_expr_(true) {
    sub_vec_->push_back(this);
    tree_ = ParseExpr();
    if (!SubCheck()) throw exception_t("expr '" + expr_ + "', bad parenthesis count");
  }

  ~Expr() {
    if (!is_sub_expr_) {
      delete cache_;
      for (auto it : *sub_vec_) delete it;
      delete sub_vec_;
      delete var_vec_;
      delete tree_;
    }
  }

  std::string GetStr() const { return expr_; }
  const expr_cache_t* GetCache() const { return cache_; }
  const bin_expr_t* GetTree() const { return tree_; }

  args_t Eval(const args_cache_t& args) const {
    args_t result = 0;
    try {
      result = tree_->Eval(args);
    } catch (const div_zero_exception_t& e) {
      if (div_zero_exc_on)
        std::cout << "Expr::Eval() exc(" << e.what() << ") : " << String() << std::endl;
    } catch (const exception_t& e) {
      throw e;
    }
    return result;
  }

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
    unsigned open_n = 0;
    while (i < expr_.length()) {
      switch (Symb(i)) {
        case '*':
        case '/':
        case '+':
        case '-':
          goto end;
        case '(':
          ++open_n;
          break;
        case ')':
          if (open_n != 0) i += 1;
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

  static const bool div_zero_exc_on = false;

  const std::string expr_;
  unsigned pos_;
  unsigned sub_count_;
  const bin_expr_t* tree_;
  const expr_cache_t* const cache_;
  std::vector<const Expr*>* sub_vec_;
  std::vector<std::string>* var_vec_;
  const bool is_sub_expr_;
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
  args_t Eval(const args_cache_t& args) const {
    const args_t denominator = arg2_->Eval(args);
    if (denominator == 0) throw div_zero_exception_t("div_expr_t::Eval()");
    return (static_cast<double>(arg1_->Eval(args)) / denominator);
  }
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

class fun_expr_t : public bin_expr_t {
 public:
  typedef std::vector<var_expr_t> vvect_t;
  fun_expr_t(const std::string& fname, const std::string& vname, const uint32_t& vnum)
      : fname_(fname) {
    for (uint32_t i = 0; i < vnum; ++i) {
      std::ostringstream var_full_name;
      var_full_name << vname << "[" << i << "]";
      vvect.push_back(var_expr_t(var_full_name.str()));
    }
  }
  const vvect_t& GetVars() const { return vvect; }
  std::string Symbol() const {
    const std::string var = vvect[0].Symbol();
    const std::string vname = var.substr(0, var.length() - 3);
    std::ostringstream oss;
    std::string str("(");
    str.back() = ')';
    oss << fname_ << "(" << vname << "," << vvect.size() << ")";
    return oss.str();
  }

 private:
  const std::string fname_;
  vvect_t vvect;
};
class sum_expr_t : public fun_expr_t {
 public:
  sum_expr_t(const std::string& vname, const uint32_t& vnum) : fun_expr_t("sum", vname, vnum) {}
  args_t Eval(const args_cache_t& args) const {
    args_t result = 0;
    for (const auto& var : GetVars()) result += var.Eval(args);
    return result;
  }
};
class avr_expr_t : public fun_expr_t {
 public:
  avr_expr_t(const std::string& vname, const uint32_t& vnum) : fun_expr_t("avr", vname, vnum) {}
  args_t Eval(const args_cache_t& args) const {
    args_t result = 0;
    for (const auto& var : GetVars()) result += var.Eval(args);
    return result / GetVars().size();
  }
};
class min_expr_t : public fun_expr_t {
 public:
  min_expr_t(const std::string& vname, const uint32_t& vnum) : fun_expr_t("min", vname, vnum) {}
  args_t Eval(const args_cache_t& args) const {
    args_t result = ARGS_MAX;
    for (const auto& var : GetVars()) {
      args_t val = var.Eval(args);
      result = (val < result) ? val : result;
    }
    return result;
  }
};
class max_expr_t : public fun_expr_t {
 public:
  max_expr_t(const std::string& vname, const uint32_t& vnum) : fun_expr_t("max", vname, vnum) {}
  args_t Eval(const args_cache_t& args) const {
    args_t result = 0;
    for (const auto& var : GetVars()) {
      args_t val = var.Eval(args);
      result = (val > result) ? val : result;
    }
    return result;
  }
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
  }

  if (arg == NULL) {
    const std::size_t pos = str.find('(');
    if (pos != std::string::npos) {
      char* fname = NULL;
      char* vname = NULL;
      int vnum = 0;
      int ret = sscanf(str.c_str(), "%m[a-zA-Z_](%m[0-9a-zA-Z_],%d)", &fname, &vname, &vnum);
      if (ret == 3) {
        const std::string fun_name(fname);
        const fun_expr_t* farg = NULL;
        if (fun_name == "sum") {
          farg = new sum_expr_t(vname, vnum);
        } else if (fun_name == "avr") {
          farg = new avr_expr_t(vname, vnum);
        } else if (fun_name == "min") {
          farg = new min_expr_t(vname, vnum);
        } else if (fun_name == "max") {
          farg = new max_expr_t(vname, vnum);
        }
        if (farg)
          for (const auto& var : farg->GetVars()) obj->AddVar(var.Symbol());
        arg = farg;
      }
      free(fname);
      free(vname);
    }
  }

  if (arg == NULL) {
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

#endif  // _SRC_CORE_COUNTERS_METRICS_XML_EXPR_H
