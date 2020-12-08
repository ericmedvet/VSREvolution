package it.units.erallab;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public interface RealFunction extends Function<double[], double[]> {
  int getInputDim();

  int getOutputDim();

  static RealFunction from(int inputDim, int outputDim, Function<double[], double[]> f) {
    return new DefaultRealFunction(inputDim, outputDim, f);
  }

  class DefaultRealFunction implements RealFunction {
    @JsonProperty
    private final int inputDim;
    @JsonProperty
    private final int outputDim;
    @JsonProperty
    private final Function<double[], double[]> function;

    @JsonCreator
    public DefaultRealFunction(
        @JsonProperty("inputDim")
            int inputDim,
        @JsonProperty("outputDim")
            int outputDim,
        @JsonProperty("function")
            Function<double[], double[]> function
    ) {
      this.inputDim = inputDim;
      this.outputDim = outputDim;
      this.function = function;
    }

    @Override
    public int getInputDim() {
      return inputDim;
    }

    @Override
    public int getOutputDim() {
      return outputDim;
    }

    @Override
    public double[] apply(double[] input) {
      return function.apply(input);
    }

    @Override
    public String toString() {
      return String.format("f:R^%d->R^%d = %s", inputDim, outputDim, function);
    }
  }
}
