"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import operator
import types
import typing as tp


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    BINARY_OPERATORS = {
        8: operator.pow,
        5: operator.mul,
        2: operator.floordiv,
        11: operator.truediv,
        6: operator.mod,
        0: operator.add,
        10: operator.sub,
        13: operator.iadd,
        23: operator.isub,
        18: operator.imul,
        24: operator.itruediv,
        15: operator.ifloordiv,
        19: operator.imod,
        21: operator.ipow,
        16: operator.ilshift,
        22: operator.irshift,
        14: operator.iand,
        20: operator.ior,
        25: operator.ixor,
        3: operator.lshift,
        9: operator.rshift,
        1: operator.and_,
        12: operator.xor,
        7: operator.or_,
    }

    COMPARE_OPERATORS = {
        0: operator.lt,
        1: operator.le,
        2: operator.eq,
        3: operator.ne,
        4: operator.gt,
        5: operator.ge,
        6: lambda x, y: x in y,
        7: lambda x, y: x not in y,
        8: lambda x, y: x is y,
        9: lambda x, y: x is not y,
        10: lambda x, y: issubclass(x, Exception) and issubclass(x, y)
    }

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.ops_generator = dis.get_instructions(self.code)
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.was_instruction: list[tp.Any] = []
        self.pushed_null: list[int] = []
        self.was_index = 0
        self.need_skip = 0
        self.offset2index: dict[int, int] = {}
        self.was_kw = False
        self.kw_skip = 0
        self.kw_values: dict[str, tp.Any] = {}
        self.wait = -1
        self.finised = False
        self.return_value = None

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        if len(self.pushed_null) and len(self.data_stack) == self.pushed_null[-1]:
            self.pushed_null.pop()
            self.data_stack.pop()
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        returned = []
        for _ in range(n):
            returned.append(self.pop())
        returned.reverse()
        return returned

    def run(self) -> tp.Any:
        while True:
            try:
                if self.was_index == len(self.was_instruction):
                    instruction = self.ops_generator.__next__()
                    if self.need_skip and instruction.offset < self.wait:
                        pass
                    else:
                        self.need_skip = False
                        getattr(self, instruction.opname.lower() + "_op")(instruction.argval, instruction.arg)
                        if 'KW_NAMES' == instruction.opname:
                            self.was_kw = True
                            self.kw_skip = len(self.code.co_consts[tp.cast(int, instruction.arg)])
                        elif 'PRECALL' != instruction.opname:
                            self.was_kw = False
                            self.kw_values.clear()
                    self.offset2index[instruction.offset] = len(self.was_instruction)
                    self.was_instruction.append(instruction)
                else:
                    instruction = self.was_instruction[self.was_index]
                    if self.need_skip and self.was_index < self.wait:
                        pass
                    else:
                        getattr(self, instruction.opname.lower() + "_op")(instruction.argval, instruction.arg)
                        if 'KW_NAMES' == instruction.opname:
                            self.was_kw = True
                            self.kw_skip = len(self.code.co_consts[tp.cast(int, instruction.arg)])
                        elif 'PRECALL' != instruction.opname:
                            self.was_kw = False
                            self.kw_values.clear()
                self.was_index += 1
                if instruction.opname == 'RETURN_GENERATOR' and not self.need_skip:
                    def f(*_: tuple[tp.Any, ...], **__: dict[str, tp.Any]) -> tp.Any:
                        fr = Frame(self.code, self.builtins, self.globals, self.locals)
                        fr.ops_generator.__next__()
                        fr.ops_generator.__next__()
                        while True:
                            res = fr.run()
                            if not fr.finised:
                                yield res
                            else:
                                break

                    return f()
                elif instruction.opname in ("YIELD_VALUE", "RETURN_VALUE") and not self.need_skip:
                    if instruction.opname == "RETURN_VALUE":
                        self.finised = True
                    break
            except StopIteration:
                self.finised = True
                break
        return self.return_value

    def resume_op(self, argval: int, arg: int) -> tp.Any:
        pass

    def push_null_op(self, argval: int, arg: int) -> tp.Any:
        self.push(None)
        self.pushed_null.append(len(self.data_stack))

    def precall_op(self, argval: int, arg: int) -> tp.Any:
        pass

    def call_op(self, argval: int, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        if not self.was_kw:
            arguments = self.popn(argval)
        else:
            arguments = self.popn(argval - self.kw_skip)
        skip = False
        if not argval:
            pot_it = None
            pot_gen = None
            try:
                pot_it = self.data_stack.pop()
            except IndexError:
                skip = True
            if not skip:
                try:
                    pot_gen = self.pop()
                except IndexError:
                    self.push(pot_it)
                    skip = True
            if not skip:
                if hasattr(pot_gen, "__call__") and hasattr(pot_it, "__next__"):
                    self.push(tp.cast(tp.Any, pot_gen)(pot_it))
                    return
                self.push(pot_gen, pot_it)
        f = self.pop()
        self.push(f(*arguments, **self.kw_values))

    def load_name_op(self, argval: str, arg: int) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        # TODO: parse all scopes
        # DONE
        if argval in self.locals:
            self.push(self.locals[argval])
        elif argval in self.globals:
            self.push(self.globals[argval])
        elif argval in self.builtins:
            self.push(self.builtins[argval])
        else:
            raise NameError

    def import_name_op(self, argval: str, arg: int) -> None:
        level, fromlist = self.popn(2)
        self.push(
            __import__(argval, self.globals, self.locals, fromlist, level)
        )

    def import_from_op(self, argval: str, arg: int) -> None:
        mod = self.top()
        self.push(getattr(mod, argval))

    def import_star_op(self, argval: None, arg: int) -> None:
        mod = self.pop()
        for attr in dir(mod):
            if attr[0] != '_':
                self.locals[attr] = getattr(mod, attr)

    def load_global_op(self, argval: str, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        # TODO: parse all scopes
        if argval in self.globals:
            self.push(self.globals[argval])
        elif argval in self.builtins:
            self.push(self.builtins[argval])
        else:
            raise NameError

    def load_const_op(self, argval: tp.Any, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(argval)

    def return_value_op(self, argval: tp.Any, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()

    def pop_top_op(self, argval: tp.Any, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def make_function_op(self, argval: int, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()  # the code associated with the function (at TOS1)
        # TODO: use arg to parse function defaults
        # DONE
        pos_def = ()
        kw_def = {}
        if arg > 1:
            kw_def = self.pop()
        if arg in (1, 3):
            pos_def = self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            # TODO: parse input arguments using code attributes such as co_argcount
            parsed_args: dict[str, tp.Any] = bind_args(code, pos_def, kw_def, *args, **kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def binary_op_op(self, argval: int, arg: int) -> None:
        x, y = self.popn(2)
        self.push(self.BINARY_OPERATORS[argval](x, y))

    def compare_op_op(self, argval: str, arg: int) -> None:
        x, y = self.popn(2)
        self.push(self.COMPARE_OPERATORS[arg](x, y))

    def get_iter_op(self, argval: None, arg: None) -> None:
        it = iter(self.pop())
        self.push(it)

    def for_iter_op(self, argval: int, arg: int) -> None:
        iterobj = self.pop()
        self.push(iterobj)
        try:
            v = next(iterobj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump(argval)

    def binary_subscr_op(self, argval: int, arg: int) -> None:
        x, y = self.popn(2)
        self.push(operator.getitem(x, y))

    def unary_negative_op(self, argval: tp.Any, arg: int) -> None:
        self.data_stack[-1] *= -1

    def unary_invert_op(self, argval: None, arg: None) -> None:
        self.data_stack[-1] = ~self.data_stack[-1]

    def unary_not_op(self, argval: None, arg: None) -> None:
        self.data_stack[-1] = not self.data_stack[-1]

    def store_name_op(self, argval: str, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[argval] = const

    def store_global_op(self, argval: str, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.globals[argval] = const

    def store_attr_op(self, argval: str, arg: int) -> None:
        val, obj = self.popn(2)
        setattr(obj, argval, val)

    def load_method_op(self, argval: tp.Any, arg: int) -> None:
        obj = self.pop()
        if callable(getattr(obj, argval)):
            self.push(obj.__getattribute__(argval))
        else:
            self.load_name_op(argval, arg)
            self.push_null_op(argval, arg)

    def load_attr_op(self, argval: str, arg: int) -> None:
        obj = self.pop()
        val = getattr(obj, argval)
        self.push(val)

    def load_fast_op(self, argval: str, arg: int) -> None:
        if argval in self.locals:
            val = self.locals[argval]
        else:
            raise UnboundLocalError(
                f"local variable '{argval}' referenced before assignment"
            )
        self.push(val)

    def kw_names_op(self, argval: tp.Any, arg: int) -> None:
        self.kw_values.clear()
        values = self.popn(len(self.code.co_consts[arg]))
        for k, v in zip(self.code.co_consts[arg], values):
            self.kw_values[k] = v

    def jump_backward_op(self, argval: int, arg: int) -> None:
        self.jump(argval)

    def jump_forward_op(self, argval: int, arg: int) -> None:
        self.jump(argval)

    def delete_attr_op(self, argval: str, arg: int) -> None:
        obj = self.pop()
        delattr(obj, argval)

    def delete_name_op(self, argval: str, arg: int) -> None:
        del self.locals[argval]

    def delete_fast_op(self, argval: str, arg: int) -> None:
        self.delete_name_op(argval, arg)

    def pop_jump_forward_if_false_op(self, argval: int, arg: int) -> None:
        cond = self.pop()
        if not cond:
            self.jump(argval)

    def pop_jump_forward_if_true_op(self, argval: int, arg: int) -> None:
        cond = self.pop()
        if cond:
            self.jump(argval)

    def jump_if_true_or_pop_op(self, argval: int, arg: int) -> None:
        val = self.top()
        if val:
            self.jump(argval)
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, argval: int, arg: int) -> None:
        val = self.top()
        if not val:
            self.jump(argval)
        else:
            self.pop()

    def pop_jump_forward_if_none_op(self, argval: int, arg: int) -> None:
        value = self.pop()
        if value is None:
            self.jump(argval)

    def jump(self, to_op: int) -> None:
        if to_op in self.offset2index:
            self.was_index = self.offset2index[to_op] - 1
        else:
            self.need_skip = True
            self.wait = to_op

    def is_op_op(self, argval: int, arg: int) -> None:
        x, y = self.popn(2)
        self.push(x is y if not argval else x is not y)

    def contains_op_op(self, argval: int, arg: int) -> None:
        x, y = self.popn(2)
        self.push(x in y if not argval else x not in y)

    def format_value_op(self, argval: tuple[tp.Any, bool], arg: int) -> None:
        value = self.pop()
        self.push(argval[0](value) if argval[0] is not None else str(value))

    def build_string_op(self, argval: int, arg: int) -> None:
        elements = self.popn(argval)
        cur_str = ""
        for e in elements:
            cur_str += e
        self.push(cur_str)

    def build_slice_op(self, argval: int, arg: int) -> None:
        if argval == 2:
            x, y = self.popn(2)
            self.push(slice(x, y))
        elif argval == 3:
            x, y, z = self.popn(3)
            self.push(slice(x, y, z))

    def build_list_op(self, argval: int, arg: int) -> None:
        lits_ = self.popn(argval)
        self.push(lits_)

    def list_append_op(self, argval: int, arg: int) -> None:
        val = self.pop()
        self.data_stack[-argval].append(val)

    def build_tuple_op(self, argv: int, arg: int) -> None:
        lits_ = self.popn(argv)
        self.push(tuple(lits_))

    def store_subscr_op(self, argv: None, arg: None) -> None:
        val, obj, subscr = self.popn(3)
        obj[subscr] = val

    def store_fast_op(self, argv: str, arg: int) -> None:
        self.locals[argv] = self.pop()

    def delete_subscr_op(self, argv: None, arg: None) -> None:
        obj, subscr = self.popn(2)
        del obj[subscr]

    def unpack_sequence_op(self, argval: None, arg: None) -> None:
        seq = self.pop()
        for x in reversed(seq):
            self.push(x)

    def list_extend_op(self, argval: int, arg: int) -> None:
        to_extend_el = self.pop()
        self.data_stack[-argval].extend(to_extend_el)

    def build_const_key_map_op(self, argval: int, arg: int) -> None:
        keys = self.pop()
        values = self.popn(argval)
        self.push({k: v for k, v in zip(keys, values)})

    def build_map_op(self, argval: None, arg: None) -> None:
        self.push({})

    def map_add_op(self, argval: int, arg: None) -> None:
        key, val = self.popn(2)
        self.data_stack[-argval][key] = val

    def build_set_op(self, argval: int, arg: int) -> None:
        elements = self.popn(argval)
        self.push(set(elements))

    def set_add_op(self, argval: int, arg: int) -> None:
        val = self.pop()
        self.data_stack[-argval].add(val)

    def set_update_op(self, argval: int, arg: int) -> None:
        elemenet = self.pop()
        self.data_stack[-argval].update(elemenet)

    def copy_op(self, argval: int, arg: int) -> None:
        self.push(self.data_stack[-argval])

    def swap_op(self, argval: int, arg: int) -> None:
        self.data_stack[-1], self.data_stack[-argval] = self.data_stack[-argval], self.data_stack[-1]

    def return_generator_op(self, argval: None, arg: int) -> None:
        pass

    def nop_op(self, argval: None, arg: None) -> None:
        pass

    def yield_value_op(self, argval: None, arg: None) -> None:
        self.return_value = self.top()


ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
ERR_MISSING_POS_ARGS = 'Missing positional arguments'
ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'


def bind_args(code: tp.Any, pos_def: tuple[tp.Any, ...],
              kw_def: dict[str, tp.Any], *args: tp.Any, **kwargs: tp.Any) -> dict[str, tp.Any]:
    ans = {}
    dict_with_def = {}
    variables = code.co_varnames
    positional = list(variables)[:code.co_argcount]
    it_args = 0
    is_kw = (code.co_flags & (1 << 3)) > 0
    pos_amount = len(pos_def) if pos_def is not None else 0
    index = code.co_argcount - 1
    if pos_def is not None:
        for i in range(pos_amount - 1, -1, -1):
            dict_with_def[positional[index]] = pos_def[i]
            index -= 1
    for it_pos in range(code.co_posonlyargcount):
        if positional[it_pos] in kwargs and not is_kw:
            raise TypeError(ERR_POSONLY_PASSED_AS_KW)
        if it_pos < len(args):
            ans[positional[it_pos]] = args[it_pos]
            it_args += 1
        elif positional[it_pos] in dict_with_def:
            ans[positional[it_pos]] = dict_with_def[positional[it_pos]]
        else:
            raise TypeError(ERR_MISSING_POS_ARGS)
    banned = []
    pos_in_kw = False
    for it_pos in range(code.co_posonlyargcount, len(positional)):
        if it_args < len(args):
            if positional[it_pos] in kwargs:
                raise TypeError(ERR_MULT_VALUES_FOR_ARG)
            ans[positional[it_pos]] = args[it_args]
            it_args += 1
        elif positional[it_pos] in kwargs:
            ans[positional[it_pos]] = kwargs[positional[it_pos]]
            banned.append(positional[it_pos])
            pos_in_kw = True
        elif positional[it_pos] in dict_with_def:
            ans[positional[it_pos]] = dict_with_def[positional[it_pos]]
        else:
            raise TypeError(ERR_MISSING_POS_ARGS)
    max_len = code.co_argcount + code.co_kwonlyargcount
    shift = 0
    if code.co_flags & (1 << 2):
        name = variables[max_len]
        shift = 1
        ans[name] = tuple(list(args)[it_args:])
    elif it_args < len(args):
        if pos_in_kw:
            raise TypeError(ERR_MULT_VALUES_FOR_ARG)
        raise TypeError(ERR_TOO_MANY_POS_ARGS)
    for it_pos in range(len(positional), max_len):
        if variables[it_pos] in kwargs:
            ans[variables[it_pos]] = kwargs[variables[it_pos]]
            banned.append(variables[it_pos])
        elif kw_def is not None and variables[it_pos] in kw_def:
            ans[variables[it_pos]] = kw_def[variables[it_pos]]
        else:
            raise TypeError(ERR_MISSING_KWONLY_ARGS)
    if is_kw:
        name = variables[max_len + shift]
        ans[name] = {k: kwargs[k] for k in filter(lambda x: x not in banned, kwargs.keys())}
    elif len(banned) != len(kwargs.keys()):
        raise TypeError(ERR_TOO_MANY_KW_ARGS)
    return ans


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
