### 

The negative eigenvalues appear even before I do basis change and partial tracing. 
My current guess is that there’s something wrong with using qutip.lindblad_dissipator(dressed_truncated_a) as the decay term. 

Let’s call negative eigenvalues “unphysical”. Here’s some observations on the unphysicalness of different setups:

1) When there’s no decay term. The states are totally physical, which rules out the probability that truncation of hamiltonian gives rise to unphysicalness
2) When I use the dressed_truncated_a as decay term, the unphysicalness is relatively low (the negative eigenvalues are smaller in magnitude)
3) When I use qutip.lindblad_dissipator(dressed_truncated_a) as the decay term, the unphysicalness is high. (the negative eigenvalues have similar magnitude of the positive eigenvalues)

What’s potentially wrong about lindblad_dissipator(dressed_truncated_a) is that assembling the superoperator from truncated operators is different from assembling superoperator from untruncated operators and then truncate the superoperator. This is because the superoperator envolves a matrix dot product of pre-multiplication and post-multiplication operators, so every entry of the superoperator depends on all levels.
What’s a bit complicated about assembling superoperator first and truncate later is the size. The dressed operator of a system of 30 level qubit coupled to 70 level oscillator is 2100 by 2100. The pre-multiplication and post-multiplication operators are 2100^2 by 2100^2. And I have to dot multiply the pre-multiplication and post-multiplication operators. It’s obviously not doable in dense form. It’s also not directly doable in sparse format as the pre-multiplication operator has a size of 20GB. What I’m doing is to slice the pre operator into row strips and the post operator into column strips, then I can compute the little square patched of the resulting superoperator. Finally I can slice the resulting superoperator on all four dimensions and try using that in qutip.mesolve and see if there’s still negative eigenvalues. 