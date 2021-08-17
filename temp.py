from CCGIR import Retrieval

ccgir = Retrieval()

print("Sentences to vectors")
ccgir.encode_file()

print("加载索引")
ccgir.build_index(n_list=1)
ccgir.index.nprob = 1

code = "function finalize ( ) after deadline { require ( ! crowdsale ended ) ; token reward . burn ( ) ; finalize ( token owner , amount raised ) ; crowdsale ended = BOOL_ ; }"

ast = "SourceUnit ContractDefinition FunctionDefinition Block ExpressionStatement FunctionCall Identifier UnaryOperation Identifier ExpressionStatement FunctionCall MemberAccess Identifier ExpressionStatement FunctionCall Identifier Identifier Identifier ExpressionStatement BinaryOperation Identifier BooleanLiteral ModifierInvocation"
sim_code, sim_ast, sim_nl = ccgir.single_query(code, ast, topK=5)

print(sim_code)
