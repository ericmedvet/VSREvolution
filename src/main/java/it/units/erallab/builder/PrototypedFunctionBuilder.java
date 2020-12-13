package it.units.erallab.builder;

import java.util.function.Function;

/**
 * @author eric
 */
public interface PrototypedFunctionBuilder<A, B> {
  Function<A, B> buildFor(B b);

  A exampleFor(B b);

  default <C> PrototypedFunctionBuilder<C, B> compose(PrototypedFunctionBuilder<C, A> other) {
    PrototypedFunctionBuilder<A, B> thisB = this;
    return new PrototypedFunctionBuilder<C, B>() {
      @Override
      public Function<C, B> buildFor(B b) {
        return thisB.buildFor(b).compose(other.buildFor(thisB.exampleFor(b)));
      }

      @Override
      public C exampleFor(B b) {
        return other.exampleFor(thisB.exampleFor(b));
      }
    };
  }

}
