package it.units.erallab;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import it.units.erallab.hmsrobots.core.controllers.Resettable;

import java.io.Serializable;
import java.util.function.Function;

/**
 * @author eric
 */
public class RealFunction implements Function<double[], double[]>, Resettable, Serializable {

  public static RealFunction from(int nOfInputs, int nOfOutputs, Function<double[], double[]> f) {
    return new RealFunction(nOfInputs, nOfOutputs, f);
  }

  @JsonProperty
  private final int nOfInputs;
  @JsonProperty
  private final int nOfOutputs;
  @JsonProperty
  @JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, property = "@class")
  private final Function<double[], double[]> function;

  @JsonCreator
  public RealFunction(
      @JsonProperty("nOfInputs") int nOfInputs,
      @JsonProperty("nOfOutputs") int nOfOutputs,
      @JsonProperty("function") Function<double[], double[]> function
  ) {
    this.nOfInputs = nOfInputs;
    this.nOfOutputs = nOfOutputs;
    this.function = function;
  }

  public int getNOfInputs() {
    return nOfInputs;
  }

  public int getNOfOutputs() {
    return nOfOutputs;
  }

  public double[] apply(double[] input) {
    return function.apply(input);
  }

  @Override
  public void reset() {
    if (function instanceof Resettable) {
      ((Resettable) function).reset();
    }
  }

  @Override
  public String toString() {
    return String.format("f:R^%d->R^%d = %s", nOfInputs, nOfOutputs, function);
  }
}
