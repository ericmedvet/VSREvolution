package it.units.erallab;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import it.units.erallab.hmsrobots.util.SerializableFunction;

import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public interface RealFunction extends SerializableFunction<double[], double[]> {
  int getNOfInputs();

  int getNOfOutputs();

  static RealFunction from(int nOfInputs, int nOfOutputs, Function<double[], double[]> f) {
    return new DefaultRealFunction(nOfInputs, nOfOutputs, f);
  }

  class DefaultRealFunction implements RealFunction {
    @JsonProperty
    private final int nOfInputs;
    @JsonProperty
    private final int nOfOutputs;
    @JsonProperty
    @JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, property = "@class")
    private final Function<double[], double[]> function;

    @JsonCreator
    public DefaultRealFunction(
        @JsonProperty("nOfInputs")
            int nOfInputs,
        @JsonProperty("nOfOutputs")
            int nOfOutputs,
        @JsonProperty("function")
            Function<double[], double[]> function
    ) {
      this.nOfInputs = nOfInputs;
      this.nOfOutputs = nOfOutputs;
      this.function = function;
    }

    @Override
    public int getNOfInputs() {
      return nOfInputs;
    }

    @Override
    public int getNOfOutputs() {
      return nOfOutputs;
    }

    @Override
    public double[] apply(double[] input) {
      return function.apply(input);
    }

    @Override
    public String toString() {
      return String.format("f:R^%d->R^%d = %s", nOfInputs, nOfOutputs, function);
    }
  }
}
